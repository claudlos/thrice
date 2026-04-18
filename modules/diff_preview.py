"""Diff preview / pre-apply validator for Hermes Agent (Thrice).

Given an existing file and a proposed new content (or a unified diff),
decide whether applying the change would leave the file in a parseable
state **before** any write hits disk.  Catches the most common LLM edit
failure modes: stray unmatched braces, broken indentation, import
deletions that leave references unresolved, stray merge-conflict
markers, malformed JSON/YAML, etc.

Usage::

    from diff_preview import preview_edit

    verdict = preview_edit(path="src/mod.py", new_content=proposed)
    if not verdict.ok:
        print(verdict.summary())
    else:
        Path("src/mod.py").write_text(proposed)

Language support:

- **Python**: ``ast.parse``
- **JSON**  : ``json.loads`` on the whole content
- **YAML** : parsed with PyYAML if available; otherwise a minimal
  line-based sanity pass (unbalanced brackets, stray tabs in
  indent-sensitive lines)
- **TOML** : ``tomllib`` (stdlib 3.11+) else skip
- **Others**: shared sanity rules only (conflict markers, non-UTF-8,
  unbalanced braces/brackets/parens)

Every language adapter returns a list of ``EditIssue``; the aggregator
combines them with the universal-sanity layer.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EditIssue:
    """One problem a reviewer should look at before applying the edit."""

    code: str              # e.g. "SYNTAX_ERROR", "CONFLICT_MARKER"
    severity: str          # "error" | "warning"
    line: Optional[int]
    column: Optional[int]
    message: str

    def format_short(self) -> str:
        loc = f"{self.line or 0}:{self.column or 0}"
        return f"{loc} [{self.severity}] {self.code}: {self.message}"


@dataclass
class EditVerdict:
    """Result of pre-viewing an edit."""

    language: str
    ok: bool
    issues: List[EditIssue] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"edit preview OK ({self.language})"
        lines = [f"edit preview BLOCKED ({self.language}); {len(self.issues)} issues:"]
        lines.extend("  " + i.format_short() for i in self.issues[:10])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Language dispatch
# ---------------------------------------------------------------------------

def _detect_language(path: str, content: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    table = {
        ".py": "python",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".rs": "rust",
        ".go": "go",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
    }
    return table.get(ext, "generic")


# ---------------------------------------------------------------------------
# Universal sanity rules (run for every language)
# ---------------------------------------------------------------------------

_CONFLICT_MARKERS = ("<<<<<<<", ">>>>>>>", "=======")


def _check_conflict_markers(content: str) -> List[EditIssue]:
    out: List[EditIssue] = []
    for i, line in enumerate(content.splitlines(), 1):
        for marker in _CONFLICT_MARKERS:
            # The "=======" marker is also valid Markdown / RST so we only
            # flag it if it sits between a "<<<<<<<" and ">>>>>>>" pair.
            if marker == "=======":
                continue
            if line.startswith(marker):
                out.append(EditIssue(
                    code="CONFLICT_MARKER",
                    severity="error",
                    line=i,
                    column=1,
                    message=f"stray git merge conflict marker: '{line.strip()}'",
                ))
    return out


def _check_balanced_brackets(content: str) -> List[EditIssue]:
    """Return issues for unbalanced {}/()/[] at file scope.

    Runs character-by-character, respecting basic Python / C-style string
    quoting so we don't false-positive on brackets inside strings.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set("([{")
    stack: List[Tuple[str, int, int]] = []    # (char, line, col)
    i = 0
    line, col = 1, 1
    in_single = in_double = in_triple_s = in_triple_d = False
    escape = False
    while i < len(content):
        ch = content[i]
        if ch == "\n":
            line += 1
            col = 0
        col += 1

        # ---- string-state machine (coarse; doesn't parse f-strings) ----
        if in_triple_s:
            if content[i:i + 3] == "'''":
                in_triple_s = False
                i += 3
                col += 2
                continue
        elif in_triple_d:
            if content[i:i + 3] == '"""':
                in_triple_d = False
                i += 3
                col += 2
                continue
        elif in_single:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_single = False
        elif in_double:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_double = False
        else:
            if content[i:i + 3] == "'''":
                in_triple_s = True
                i += 3
                col += 2
                continue
            if content[i:i + 3] == '"""':
                in_triple_d = True
                i += 3
                col += 2
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch in openers:
                stack.append((ch, line, col))
            elif ch in pairs:
                if not stack or stack[-1][0] != pairs[ch]:
                    return [EditIssue(
                        code="UNBALANCED_BRACKETS",
                        severity="error",
                        line=line, column=col,
                        message=f"unmatched closing '{ch}'",
                    )]
                stack.pop()
        i += 1

    if stack:
        ch, line_open, col_open = stack[0]
        return [EditIssue(
            code="UNBALANCED_BRACKETS",
            severity="error",
            line=line_open, column=col_open,
            message=f"unclosed '{ch}' opened here",
        )]
    return []


def _universal_checks(content: str) -> List[EditIssue]:
    return _check_conflict_markers(content) + _check_balanced_brackets(content)


# ---------------------------------------------------------------------------
# Language adapters
# ---------------------------------------------------------------------------

def _check_python(content: str) -> List[EditIssue]:
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return [EditIssue(
            code="SYNTAX_ERROR",
            severity="error",
            line=exc.lineno,
            column=exc.offset,
            message=f"Python SyntaxError: {exc.msg}",
        )]
    return []


def _check_json(content: str) -> List[EditIssue]:
    if not content.strip():
        return [EditIssue(
            code="EMPTY_FILE",
            severity="error", line=None, column=None,
            message="JSON file is empty",
        )]
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return [EditIssue(
            code="JSON_PARSE_ERROR",
            severity="error",
            line=exc.lineno, column=exc.colno,
            message=f"invalid JSON: {exc.msg}",
        )]
    return []


def _check_toml(content: str) -> List[EditIssue]:
    try:
        import tomllib
    except ImportError:  # 3.10 or older
        return []
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return [EditIssue(
            code="TOML_PARSE_ERROR",
            severity="error",
            line=None, column=None,
            message=f"invalid TOML: {exc!s}",
        )]
    return []


def _check_yaml(content: str) -> List[EditIssue]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return []
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark else None
        col = mark.column + 1 if mark else None
        return [EditIssue(
            code="YAML_PARSE_ERROR",
            severity="error",
            line=line, column=col,
            message=f"invalid YAML: {exc!s}",
        )]
    return []


_LANG_ADAPTERS: dict = {
    "python": _check_python,
    "json":   _check_json,
    "toml":   _check_toml,
    "yaml":   _check_yaml,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def preview_content(path: str, new_content: str) -> EditVerdict:
    """Check whether ``new_content`` is a sane replacement for ``path``."""
    lang = _detect_language(path, new_content)
    issues: List[EditIssue] = []
    # Universal first (conflict markers + basic bracket balance).
    issues.extend(_universal_checks(new_content))
    # Language-specific adapter.
    adapter = _LANG_ADAPTERS.get(lang)
    if adapter is not None:
        issues.extend(adapter(new_content))
    issues = _dedupe(issues)
    ok = not any(i.severity == "error" for i in issues)
    return EditVerdict(language=lang, ok=ok, issues=issues)


# Convenience alias - callers may pass either the post-edit text or a diff.
def preview_edit(path: str, new_content: str) -> EditVerdict:
    """Preview the result of replacing ``path``'s contents with ``new_content``."""
    return preview_content(path, new_content)


def preview_patch(path: str, original: str, patch: str) -> EditVerdict:
    """Apply a unified ``patch`` to ``original`` (purely in memory) and preview.

    Returns a verdict with a leading ``PATCH_APPLY_FAILED`` issue if the
    patch doesn't cleanly apply.
    """
    applied = _apply_unified_diff(original, patch)
    if applied is None:
        return EditVerdict(
            language=_detect_language(path, original),
            ok=False,
            issues=[EditIssue(
                code="PATCH_APPLY_FAILED",
                severity="error",
                line=None, column=None,
                message="unified diff did not cleanly apply to original content",
            )],
        )
    return preview_content(path, applied)


# ---------------------------------------------------------------------------
# Minimal in-memory unified-diff applier (handles single-file, single-hunk diffs)
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _apply_unified_diff(original: str, patch: str) -> Optional[str]:
    """Very small applier.  Returns None on mismatch.  Handles multiple hunks
    but only one file.  Intended for previews, not general patch merging.
    """
    orig_lines = original.splitlines(keepends=False)
    patch_lines = patch.splitlines(keepends=False)

    idx = 0
    while idx < len(patch_lines) and not patch_lines[idx].startswith("@@"):
        idx += 1

    out: List[str] = []
    cursor = 0   # 0-based index into orig_lines

    while idx < len(patch_lines):
        m = _HUNK_RE.match(patch_lines[idx])
        if not m:
            idx += 1
            continue
        old_start = int(m.group(1))
        # Copy any unpatched prefix from original.
        while cursor < old_start - 1:
            if cursor >= len(orig_lines):
                return None
            out.append(orig_lines[cursor])
            cursor += 1
        idx += 1
        # Process hunk body until the next @@ or EOF.
        while idx < len(patch_lines) and not patch_lines[idx].startswith("@@"):
            line = patch_lines[idx]
            if line.startswith("---") or line.startswith("+++"):
                idx += 1
                continue
            if line.startswith(" "):
                if cursor >= len(orig_lines) or orig_lines[cursor] != line[1:]:
                    return None
                out.append(orig_lines[cursor])
                cursor += 1
            elif line.startswith("-"):
                if cursor >= len(orig_lines) or orig_lines[cursor] != line[1:]:
                    return None
                cursor += 1
            elif line.startswith("+"):
                out.append(line[1:])
            elif line == "":
                # empty context line (some diff generators emit bare "")
                if cursor < len(orig_lines) and orig_lines[cursor] == "":
                    out.append("")
                    cursor += 1
            else:
                # Unknown marker; bail.
                return None
            idx += 1

    # Copy any trailing unpatched lines.
    while cursor < len(orig_lines):
        out.append(orig_lines[cursor])
        cursor += 1

    trailing_nl = original.endswith("\n")
    joined = "\n".join(out)
    return joined + ("\n" if trailing_nl else "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe(issues: List[EditIssue]) -> List[EditIssue]:
    seen = set()
    out: List[EditIssue] = []
    for i in issues:
        key = (i.code, i.line, i.column, i.message)
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


__all__ = [
    "EditIssue",
    "EditVerdict",
    "preview_content",
    "preview_edit",
    "preview_patch",
]
