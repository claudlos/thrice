"""
Comment-Stripped Fuzzy Match Strategy — Language-aware comment stripping.

The #1 cause of "unique match not found" errors in patch operations is
comments that differ between the model's memory and the actual file.

This module provides:
  1. Language-aware comment stripping for 30+ file extensions
  2. A match_ignoring_comments() function for fuzzy matching with comments stripped

Integration point: tools/fuzzy_match.py — add as a strategy in the
fuzzy matching pipeline, after exact match and before aggressive strategies.

Usage:
    from new_files.comment_strip_matcher import strip_comments, match_ignoring_comments

    clean = strip_comments(code, language="python")
    result = match_ignoring_comments(needle, haystack, language="javascript")
"""

import re
from typing import Optional, Tuple

# ─── Comment pattern definitions ────────────────────────────────────────────
_PATTERNS = {
    # Single-line: # to end of line (Python, Ruby, Shell, YAML, etc.)
    "hash": re.compile(r"#[^\n]*"),
    # Single-line: // to end of line (C, C++, Java, JS, TS, Go, Rust, etc.)
    "double_slash": re.compile(r"//[^\n]*"),
    # Block: /* ... */ (C-family, CSS, SQL)
    "block": re.compile(r"/\*.*?\*/", re.DOTALL),
    # HTML/XML: <!-- ... -->
    "html": re.compile(r"<!--.*?-->", re.DOTALL),
    # SQL/Lua: -- to end of line
    "double_dash": re.compile(r"--[^\n]*"),
    # Lisp/Clojure: ; to end of line
    "semicolon": re.compile(r";[^\n]*"),
    # Haskell: {- ... -} block
    "haskell_block": re.compile(r"\{-.*?-\}", re.DOTALL),
    # Haskell: -- to end of line
    "haskell_line": re.compile(r"--[^\n]*"),
    # PowerShell: <# ... #> block
    "ps_block": re.compile(r"<#.*?#>", re.DOTALL),
    # Vim: " to end of line
    "vim": re.compile(r'(?:^|(?<=\s))"[^\n]*', re.MULTILINE),
    # Matlab/Octave: % to end of line
    "percent": re.compile(r"%[^\n]*"),
    # Erlang/LaTeX: % to end of line (same as matlab)
    # Fortran: ! to end of line
    "fortran": re.compile(r"![^\n]*"),
    # JSX/TSX: {/* ... */}
    "jsx_block": re.compile(r"\{/\*.*?\*/\}", re.DOTALL),
    # Python docstrings (triple quotes) — treated as comments for matching
    "triple_double": re.compile(r'""".*?"""', re.DOTALL),
    "triple_single": re.compile(r"'''.*?'''", re.DOTALL),
}

# ─── Language / extension -> comment style mapping (30+ extensions) ─────────
_LANGUAGE_COMMENTS: dict[str, list[str]] = {
    # Python family
    "python": ["hash", "triple_double", "triple_single"],
    "py": ["hash", "triple_double", "triple_single"],
    "pyi": ["hash", "triple_double", "triple_single"],
    "pyx": ["hash", "triple_double", "triple_single"],

    # Ruby
    "ruby": ["hash"],
    "rb": ["hash"],
    "rake": ["hash"],
    "gemspec": ["hash"],

    # Shell
    "shell": ["hash"],
    "sh": ["hash"],
    "bash": ["hash"],
    "zsh": ["hash"],
    "fish": ["hash"],
    "ksh": ["hash"],

    # Config / Data
    "yaml": ["hash"],
    "yml": ["hash"],
    "toml": ["hash"],
    "conf": ["hash"],
    "ini": ["hash"],
    "cfg": ["hash"],
    "properties": ["hash"],
    "env": ["hash"],
    "dockerfile": ["hash"],
    "makefile": ["hash"],
    "mk": ["hash"],
    "cmake": ["hash"],

    # Perl
    "perl": ["hash"],
    "pl": ["hash"],
    "pm": ["hash"],

    # R
    "r": ["hash"],

    # JavaScript / TypeScript family
    "javascript": ["double_slash", "block"],
    "js": ["double_slash", "block"],
    "jsx": ["double_slash", "block", "jsx_block"],
    "typescript": ["double_slash", "block"],
    "ts": ["double_slash", "block"],
    "tsx": ["double_slash", "block", "jsx_block"],
    "mjs": ["double_slash", "block"],
    "cjs": ["double_slash", "block"],

    # C / C++ family
    "c": ["double_slash", "block"],
    "h": ["double_slash", "block"],
    "cpp": ["double_slash", "block"],
    "hpp": ["double_slash", "block"],
    "cc": ["double_slash", "block"],
    "cxx": ["double_slash", "block"],
    "hxx": ["double_slash", "block"],
    "m": ["double_slash", "block"],  # Objective-C

    # C# / Java / Kotlin / Scala
    "java": ["double_slash", "block"],
    "cs": ["double_slash", "block"],
    "kotlin": ["double_slash", "block"],
    "kt": ["double_slash", "block"],
    "scala": ["double_slash", "block"],

    # Go / Rust / Swift / Dart
    "go": ["double_slash", "block"],
    "rust": ["double_slash", "block"],
    "rs": ["double_slash", "block"],
    "swift": ["double_slash", "block"],
    "dart": ["double_slash", "block"],

    # CSS family
    "css": ["block"],
    "scss": ["double_slash", "block"],
    "sass": ["double_slash", "block"],
    "less": ["double_slash", "block"],

    # HTML / XML / Templates
    "html": ["html"],
    "htm": ["html"],
    "xml": ["html"],
    "svg": ["html"],
    "xhtml": ["html"],
    "vue": ["html", "double_slash", "block"],
    "svelte": ["html", "double_slash", "block"],

    # SQL
    "sql": ["double_dash", "block"],
    "plsql": ["double_dash", "block"],
    "pgsql": ["double_dash", "block"],

    # Lua
    "lua": ["double_dash", "block"],

    # Haskell
    "haskell": ["haskell_line", "haskell_block"],
    "hs": ["haskell_line", "haskell_block"],

    # Lisp family
    "lisp": ["semicolon"],
    "el": ["semicolon"],
    "clojure": ["semicolon"],
    "clj": ["semicolon"],
    "cljc": ["semicolon"],
    "cljs": ["semicolon"],
    "scheme": ["semicolon"],
    "scm": ["semicolon"],

    # Matlab / Octave
    "matlab": ["percent"],
    "octave": ["percent"],

    # LaTeX
    "tex": ["percent"],
    "latex": ["percent"],
    "sty": ["percent"],

    # Erlang / Elixir
    "erl": ["percent"],
    "ex": ["hash"],
    "exs": ["hash"],

    # Fortran
    "f90": ["fortran"],
    "f95": ["fortran"],
    "f03": ["fortran"],
    "fortran": ["fortran"],

    # PowerShell
    "ps1": ["hash", "ps_block"],
    "psm1": ["hash", "ps_block"],

    # Vim
    "vim": ["vim"],

    # Proto
    "proto": ["double_slash", "block"],

    # GraphQL
    "graphql": ["hash"],
    "gql": ["hash"],
}

# ─── Extension normalization ───────────────────────────────────────────────


def _resolve_language(language: Optional[str], file_path: Optional[str] = None) -> Optional[str]:
    """Resolve a language identifier from a language name or file path.

    Args:
        language: Language name or file extension (e.g., "python", "py", ".py").
        file_path: File path to extract extension from.

    Returns:
        Normalized language key, or None if unresolvable.
    """
    if language:
        key = language.lower().strip().lstrip(".")
        if key in _LANGUAGE_COMMENTS:
            return key

    if file_path:
        if "." in file_path:
            ext = file_path.rsplit(".", 1)[1].lower()
            if ext in _LANGUAGE_COMMENTS:
                return ext

        # Check for files like Makefile, Dockerfile without extension
        basename = file_path.rsplit("/", 1)[-1].lower()
        if basename in ("makefile", "gnumakefile"):
            return "makefile"
        if basename in ("dockerfile",):
            return "dockerfile"
        if basename in (".env", ".env.local", ".env.production"):
            return "env"

    return None


def _get_patterns(language: Optional[str]) -> list[re.Pattern]:
    """Get comment patterns for a language."""
    if language and language in _LANGUAGE_COMMENTS:
        style_names = _LANGUAGE_COMMENTS[language]
        return [_PATTERNS[name] for name in style_names if name in _PATTERNS]

    # Default: hash + double_slash (covers most languages)
    return [_PATTERNS["hash"], _PATTERNS["double_slash"]]


def _replace_preserving_newlines(match: re.Match) -> str:
    """Replace comment with empty string, preserving newlines for line numbering."""
    return "\n" * match.group(0).count("\n")


# ─── Public API ─────────────────────────────────────────────────────────────


def strip_comments(
    text: str,
    language: Optional[str] = None,
    file_path: Optional[str] = None,
) -> str:
    """Strip comments from source text based on language.

    Preserves line count by replacing block comments with equivalent newlines.
    This is a best-effort heuristic, not a full parser — it does not handle
    comments inside string literals.

    Args:
        text: Source code text.
        language: Language name or extension (e.g., "python", "js", ".tsx").
        file_path: File path for language auto-detection.

    Returns:
        Text with comments removed (line count preserved).
    """
    resolved = _resolve_language(language, file_path)
    patterns = _get_patterns(resolved)

    result = text
    for pattern in patterns:
        result = pattern.sub(_replace_preserving_newlines, result)

    return result


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace within each line."""
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def match_ignoring_comments(
    needle: str,
    haystack: str,
    language: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """Try to find needle in haystack with comments stripped from both.

    Both needle and haystack have comments removed and whitespace normalized
    before comparison. Returns line positions in the ORIGINAL haystack.

    Args:
        needle: The text to search for.
        haystack: The text to search in (typically file content).
        language: Language name or extension.
        file_path: File path for language auto-detection.

    Returns:
        Tuple of (start_line, end_line) 0-indexed in the original haystack,
        or None if no unique match found.
    """
    resolved = _resolve_language(language, file_path)

    stripped_needle = strip_comments(needle, resolved)
    stripped_haystack = strip_comments(haystack, resolved)

    norm_needle = _normalize_whitespace(stripped_needle).strip()
    norm_haystack = _normalize_whitespace(stripped_haystack)

    if not norm_needle:
        return None

    # Non-empty lines from the needle
    needle_lines = [l for l in norm_needle.splitlines() if l.strip()]
    haystack_lines = norm_haystack.splitlines()

    if not needle_lines:
        return None

    # Sliding window search
    window_size = len(needle_lines)
    matches: list[int] = []

    for i in range(len(haystack_lines) - window_size + 1):
        window = [l.strip() for l in haystack_lines[i:i + window_size] if l.strip()]

        if len(window) != len(needle_lines):
            continue

        if all(w == n for w, n in zip(window, needle_lines)):
            matches.append(i)

    if len(matches) != 1:
        return None  # 0 or 2+ matches

    start_line = matches[0]

    # Walk the original haystack to find the true end line
    original_lines = haystack.splitlines()
    needle_idx = 0
    end_line = start_line
    for j in range(start_line, len(original_lines)):
        norm_line = _normalize_whitespace(
            strip_comments(original_lines[j], resolved)
        ).strip()
        if not norm_line:
            end_line = j
            continue
        if needle_idx < len(needle_lines):
            if norm_line == needle_lines[needle_idx].strip():
                needle_idx += 1
                end_line = j
        if needle_idx >= len(needle_lines):
            break

    return (start_line, end_line)


def get_supported_languages() -> list[str]:
    """Return all supported language keys."""
    return sorted(_LANGUAGE_COMMENTS.keys())
