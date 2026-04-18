"""
Silent Error Audit — Thrice Improvement #12

Detects and fixes silent error swallowing patterns (except: pass, broad except
with no logging, etc.) and provides a health dashboard for all Thrice modules.

Pure Python, no external dependencies. Uses regex + AST analysis.
"""

from __future__ import annotations

import ast
import difflib
import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SilentErrorPattern:
    """A detected silent-error-swallowing pattern."""
    file_path: str
    line_number: int
    pattern_type: str          # bare_except | except_pass | broad_except_pass
    code_snippet: str
    suggestion: str


@dataclass
class ModuleHealth:
    """Health status of a single Thrice module."""
    module_name: str
    importable: bool
    import_error: Optional[str] = None
    version: Optional[str] = None
    has_tests: bool = False
    test_count: int = 0


# ---------------------------------------------------------------------------
# SilentErrorScanner
# ---------------------------------------------------------------------------

class SilentErrorScanner:
    """Scan Python source for silent error swallowing patterns."""

    # Regex patterns for quick pre-screening (line-level)
    _RE_BARE_EXCEPT = re.compile(r'^\s*except\s*:\s*$')
    _RE_EXCEPT_PASS = re.compile(r'^\s*except\s*.*:\s*$')
    _RE_PASS_LINE = re.compile(r'^\s*pass\s*$')
    _RE_ELLIPSIS_LINE = re.compile(r'^\s*\.\.\.\s*$')

    # --------------- public API ---------------

    def scan_file(self, file_path: str) -> List[SilentErrorPattern]:
        """Scan a single Python file and return detected patterns."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return []
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        patterns: List[SilentErrorPattern] = []
        # AST-based detection (primary)
        patterns.extend(self._scan_ast(source, str(path)))
        # Regex fallback for things AST might miss (e.g. syntax-error files)
        if not patterns:
            patterns.extend(self._scan_regex(source, str(path)))
        return patterns

    def scan_directory(
        self, directory: str, glob_pattern: str = "*.py"
    ) -> List[SilentErrorPattern]:
        """Recursively scan a directory for silent error patterns."""
        results: List[SilentErrorPattern] = []
        root = Path(directory)
        if not root.is_dir():
            return results
        for py_file in sorted(root.rglob(glob_pattern)):
            if py_file.is_file():
                results.extend(self.scan_file(str(py_file)))
        return results

    @staticmethod
    def generate_report(patterns: List[SilentErrorPattern]) -> str:
        """Generate a human-readable report from detected patterns."""
        if not patterns:
            return "No silent error patterns detected. Code looks clean!"
        lines = [
            f"Silent Error Audit Report — {len(patterns)} pattern(s) found",
            "=" * 60,
        ]
        by_type: Dict[str, int] = {}
        for p in patterns:
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1

        lines.append("")
        lines.append("Summary by type:")
        for ptype, count in sorted(by_type.items()):
            lines.append(f"  {ptype}: {count}")
        lines.append("")
        lines.append("-" * 60)

        for i, p in enumerate(patterns, 1):
            lines.append(f"\n[{i}] {p.pattern_type} at {p.file_path}:{p.line_number}")
            lines.append(f"    Snippet : {p.code_snippet.strip()}")
            lines.append(f"    Fix     : {p.suggestion}")
        lines.append("")
        return "\n".join(lines)

    # --------------- internals ---------------

    def _scan_ast(self, source: str, file_path: str) -> List[SilentErrorPattern]:
        """Use AST to find silent error handlers."""
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return []
        source_lines = source.splitlines()
        patterns: List[SilentErrorPattern] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            handler: ast.ExceptHandler = node
            body = handler.body

            # Determine if handler body is effectively empty / just pass / ...
            is_silent = self._is_silent_body(body)
            if not is_silent:
                # Check for broad except with no logging
                if self._is_broad_except_no_logging(handler):
                    snippet = self._get_snippet(source_lines, handler.lineno)
                    patterns.append(SilentErrorPattern(
                        file_path=file_path,
                        line_number=handler.lineno,
                        pattern_type="broad_except_no_log",
                        code_snippet=snippet,
                        suggestion="Add logging (e.g. logger.exception()) to broad except block.",
                    ))
                continue

            # Classify the pattern
            lineno = handler.lineno
            snippet = self._get_snippet(source_lines, lineno)

            if handler.type is None:
                # bare except
                patterns.append(SilentErrorPattern(
                    file_path=file_path,
                    line_number=lineno,
                    pattern_type="bare_except",
                    code_snippet=snippet,
                    suggestion="Replace bare 'except: pass' with specific exception and logging.",
                ))
            else:
                exc_name = self._exception_name(handler.type)
                if exc_name in ("Exception", "BaseException"):
                    patterns.append(SilentErrorPattern(
                        file_path=file_path,
                        line_number=lineno,
                        pattern_type="broad_except_pass",
                        code_snippet=snippet,
                        suggestion=f"Catch a specific exception instead of {exc_name}, and add logging.",
                    ))
                else:
                    patterns.append(SilentErrorPattern(
                        file_path=file_path,
                        line_number=lineno,
                        pattern_type="except_pass",
                        code_snippet=snippet,
                        suggestion="Add logging or a comment explaining why the error is intentionally ignored.",
                    ))
        return patterns

    def _scan_regex(self, source: str, file_path: str) -> List[SilentErrorPattern]:
        """Regex fallback for files that fail AST parsing."""
        patterns: List[SilentErrorPattern] = []
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if self._RE_BARE_EXCEPT.match(line):
                # Check if next non-blank line is pass or ...
                nxt = self._next_nonblank(lines, i + 1)
                if nxt is not None and (
                    self._RE_PASS_LINE.match(nxt) or self._RE_ELLIPSIS_LINE.match(nxt)
                ):
                    patterns.append(SilentErrorPattern(
                        file_path=file_path,
                        line_number=i + 1,
                        pattern_type="bare_except",
                        code_snippet=line.strip(),
                        suggestion="Replace bare 'except: pass' with specific exception and logging.",
                    ))
            elif self._RE_EXCEPT_PASS.match(line):
                nxt = self._next_nonblank(lines, i + 1)
                if nxt is not None and self._RE_PASS_LINE.match(nxt):
                    patterns.append(SilentErrorPattern(
                        file_path=file_path,
                        line_number=i + 1,
                        pattern_type="except_pass",
                        code_snippet=line.strip(),
                        suggestion="Add logging or handle the exception properly.",
                    ))
        return patterns

    # --- helpers ---

    @staticmethod
    def _is_silent_body(body: List[ast.stmt]) -> bool:
        """Return True if the handler body is just pass / Ellipsis / a string."""
        if not body:
            return True
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Expr):
                val = stmt.value
                if isinstance(val, ast.Constant) and val.value is ...:
                    return True
                # docstring-only handler
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    return True
        return False

    @staticmethod
    def _is_broad_except_no_logging(handler: ast.ExceptHandler) -> bool:
        """Broad except (Exception/BaseException) with no logging call."""
        if handler.type is None:
            return False
        name = SilentErrorScanner._exception_name(handler.type)
        if name not in ("Exception", "BaseException"):
            return False
        # Look for any call to logging / logger / print / raise in the body
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
            if isinstance(node, ast.Raise):
                return False
            if isinstance(node, ast.Call):
                func = node.func
                # logger.xxx(...)  or  logging.xxx(...)
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        if func.value.id in ("logger", "logging", "log"):
                            return False
                # print(...)
                if isinstance(func, ast.Name) and func.id == "print":
                    return False
        return True

    @staticmethod
    def _exception_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @staticmethod
    def _get_snippet(lines: List[str], lineno: int, context: int = 2) -> str:
        start = max(0, lineno - 1)
        end = min(len(lines), lineno + context)
        return "\n".join(lines[start:end])

    @staticmethod
    def _next_nonblank(lines: List[str], start: int) -> Optional[str]:
        for i in range(start, min(start + 5, len(lines))):
            if lines[i].strip():
                return lines[i]
        return None


# ---------------------------------------------------------------------------
# HealthDashboard
# ---------------------------------------------------------------------------

class HealthDashboard:
    """Health dashboard for all Thrice modules."""

    THRICE_MODULES: List[str] = [
        "adaptive_compression",
        "agent_loop_shadow",
        "agent_loop_state_machine",
        "auto_commit",
        "comment_strip_matcher",
        "consensus_protocol",
        "context_mentions",
        "context_optimizer",
        "debugging_guidance",
        "enforcement",
        "hermes_invariants",
        "invariants_unified",
        "prompt_algebra",
        "repo_map",
        "self_improving_tests",
        "self_reflection",
        "skill_verifier",
        "smart_truncation",
        "state_machine",
        "structured_errors",
        "test_lint_loop",
        "token_budgeting",
        "tool_alias_map",
        "tool_chain_analysis",
        "tool_selection_model",
        "verified_messages",
        "reproduce_first",
        "project_memory",
        "conversation_checkpoint",
        "edit_format",
        "error_recovery",
        "tool_call_batcher",
        "test_fix_loop",
        "context_gatherer",
        "change_impact",
        "agent_loop_components",
        "silent_error_audit",
    ]

    def __init__(self, hermes_dir: str) -> None:
        self.hermes_dir = hermes_dir
        self._results: List[ModuleHealth] = []

    def check_module(self, module_name: str) -> ModuleHealth:
        """Check importability and test coverage for a single module."""
        health = ModuleHealth(module_name=module_name, importable=False)

        # Try importing
        try:
            mod = importlib.import_module(module_name)
            health.importable = True
            health.import_error = None
            health.version = getattr(mod, "__version__", getattr(mod, "VERSION", None))
        except Exception as exc:
            health.importable = False
            health.import_error = f"{type(exc).__name__}: {exc}"

        # Check for test file
        tests_dir = os.path.join(self.hermes_dir, "tests")
        test_file = os.path.join(tests_dir, f"test_{module_name}.py")
        if os.path.isfile(test_file):
            health.has_tests = True
            health.test_count = self._count_tests(test_file)
        else:
            # Also check in the hermes_dir root
            alt = os.path.join(self.hermes_dir, f"test_{module_name}.py")
            if os.path.isfile(alt):
                health.has_tests = True
                health.test_count = self._count_tests(alt)

        return health

    def check_all_thrice_modules(self) -> List[ModuleHealth]:
        """Check all known Thrice modules."""
        self._results = [self.check_module(m) for m in self.THRICE_MODULES]
        return list(self._results)

    def get_active_modules(self) -> List[str]:
        """Return names of modules that import successfully."""
        if not self._results:
            self.check_all_thrice_modules()
        return [m.module_name for m in self._results if m.importable]

    def get_failed_modules(self) -> List[str]:
        """Return names of modules that fail to import."""
        if not self._results:
            self.check_all_thrice_modules()
        return [m.module_name for m in self._results if not m.importable]

    def format_dashboard(self) -> str:
        """Format a human-readable health dashboard."""
        if not self._results:
            self.check_all_thrice_modules()

        active = [m for m in self._results if m.importable]
        failed = [m for m in self._results if not m.importable]
        tested = [m for m in self._results if m.has_tests]

        lines = [
            "Thrice Module Health Dashboard",
            "=" * 50,
            f"Total modules : {len(self._results)}",
            f"Importable    : {len(active)}",
            f"Failed        : {len(failed)}",
            f"With tests    : {len(tested)}",
            "",
        ]

        if active:
            lines.append("Active modules:")
            for m in active:
                ver = f" v{m.version}" if m.version else ""
                tst = f" ({m.test_count} tests)" if m.has_tests else " (no tests)"
                lines.append(f"  [OK] {m.module_name}{ver}{tst}")
            lines.append("")

        if failed:
            lines.append("Failed modules:")
            for m in failed:
                tst = f" ({m.test_count} tests)" if m.has_tests else ""
                lines.append(f"  [FAIL] {m.module_name} — {m.import_error}{tst}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _count_tests(test_file: str) -> int:
        """Count test functions/methods in a test file."""
        count = 0
        try:
            with open(test_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("def test_") or stripped.startswith("async def test_"):
                        count += 1
        except OSError:
            pass
        return count


# ---------------------------------------------------------------------------
# LoggingFixer
# ---------------------------------------------------------------------------

class LoggingFixer:
    """Suggest and apply fixes for silent error patterns."""

    _INDENT_RE = re.compile(r'^(\s*)')

    def suggest_fix(self, pattern: SilentErrorPattern) -> str:
        """Return replacement code for a single pattern."""
        indent = self._detect_indent(pattern.code_snippet)
        body_indent = indent + "    "

        if pattern.pattern_type == "bare_except":
            return (
                f"{indent}except Exception as exc:\n"
                f"{body_indent}logger.exception(\"Unexpected error: %s\", exc)\n"
                f"{body_indent}raise\n"
            )
        elif pattern.pattern_type == "broad_except_pass":
            return (
                f"{indent}except Exception as exc:\n"
                f"{body_indent}logger.exception(\"Error caught: %s\", exc)\n"
            )
        elif pattern.pattern_type == "except_pass":
            # Keep the original exception type
            exc_type = self._extract_exception_type(pattern.code_snippet)
            return (
                f"{indent}except {exc_type} as exc:\n"
                f"{body_indent}logger.debug(\"Suppressed {exc_type}: %s\", exc)\n"
            )
        elif pattern.pattern_type == "broad_except_no_log":
            return (
                f"{indent}# TODO: add logger.exception() call in this handler\n"
            )
        else:
            return f"{indent}# TODO: review this exception handler\n"

    def generate_patch(self, patterns: List[SilentErrorPattern]) -> str:
        """Generate a unified-diff style patch for all patterns."""
        if not patterns:
            return ""
        patches: List[str] = []
        # Group by file
        by_file: Dict[str, List[SilentErrorPattern]] = {}
        for p in patterns:
            by_file.setdefault(p.file_path, []).append(p)

        for fpath, file_patterns in sorted(by_file.items()):
            try:
                original = Path(fpath).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            modified = original
            # Apply fixes in reverse line order to preserve line numbers
            for p in sorted(file_patterns, key=lambda x: x.line_number, reverse=True):
                modified = self.fix_pattern(modified, p)

            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{fpath}",
                tofile=f"b/{fpath}",
            )
            patches.append("".join(diff))

        return "\n".join(patches)

    def fix_pattern(self, code: str, pattern: SilentErrorPattern) -> str:
        """Apply a single pattern fix to source code, returning modified code."""
        lines = code.splitlines(keepends=True)
        idx = pattern.line_number - 1  # 0-based
        if idx < 0 or idx >= len(lines):
            return code

        fix = self.suggest_fix(pattern)
        fix_lines = fix.splitlines(keepends=True)
        # Ensure trailing newlines
        fix_lines = [ln if ln.endswith("\n") else ln + "\n" for ln in fix_lines]

        # Determine how many original lines to replace
        # Replace the except line + the body (pass / ...)
        end = idx + 1
        if end < len(lines):
            # Skip subsequent pass / ... / docstring-only lines
            while end < len(lines):
                stripped = lines[end].strip()
                if stripped in ("pass", "...", '"""', "'''", ""):
                    end += 1
                else:
                    break

        lines[idx:end] = fix_lines
        return "".join(lines)

    # --- helpers ---

    def _detect_indent(self, snippet: str) -> str:
        first_line = snippet.split("\n")[0]
        m = self._INDENT_RE.match(first_line)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_exception_type(snippet: str) -> str:
        m = re.search(r'except\s+(\w+)', snippet)
        return m.group(1) if m else "Exception"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_health_check(hermes_dir: str) -> str:
    """One-line summary of Thrice module health."""
    dash = HealthDashboard(hermes_dir)
    results = dash.check_all_thrice_modules()
    ok = sum(1 for m in results if m.importable)
    fail = len(results) - ok
    tested = sum(1 for m in results if m.has_tests)
    return f"Thrice health: {ok}/{len(results)} importable, {fail} failed, {tested} with tests"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Simple CLI for silent error auditing."""
    import argparse

    parser = argparse.ArgumentParser(description="Silent Error Audit for Thrice/Hermes")
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Scan files for silent error patterns")
    scan_p.add_argument("path", help="File or directory to scan")

    health_p = sub.add_parser("health", help="Show Thrice module health dashboard")
    health_p.add_argument("--dir", default=".", help="Hermes project directory")

    args = parser.parse_args()

    if args.command == "scan":
        scanner = SilentErrorScanner()
        target = Path(args.path)
        if target.is_file():
            patterns = scanner.scan_file(str(target))
        else:
            patterns = scanner.scan_directory(str(target))
        print(scanner.generate_report(patterns))
    elif args.command == "health":
        print(quick_health_check(args.dir))
        dash = HealthDashboard(args.dir)
        print(dash.format_dashboard())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
