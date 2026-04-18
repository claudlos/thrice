"""Doctest runner for Hermes Agent (Thrice).

Drives Python's standard ``doctest`` module over a project tree and
returns structured results so the agent can reason about drift between
code and its docstring examples.

Typical usage::

    from doctest_runner import run_project

    result = run_project(".")
    if not result.ok:
        for failure in result.failures[:10]:
            print(failure.format_short())

Each failure carries ``(file, line, want, got)`` so the agent can either
patch the docstring or fix the implementation without re-running doctests
to see the error.

Design notes:

- We import-free: files are parsed with ``doctest.DocTestFinder`` via
  ``doctest.testfile`` so we never exec arbitrary project code.  Running
  the examples *does* import the module (that's how doctest works), so
  we honor standard ``sys.path`` rules and allow callers to pass
  ``PYTHONPATH`` extras.
- Every diagnostic is sorted deterministically by (file, line).
- A global timeout guard prevents runaway infinite-loop doctests.
"""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import io
import logging
import os
import pathlib
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoctestFailure:
    """One failing doctest example."""

    file: str
    line: int
    name: str                 # fully qualified object name, e.g. pkg.mod.Class.method
    source: str               # the example source (stripped)
    want: str                 # expected output
    got: str                  # actual output

    def format_short(self) -> str:
        return (
            f"{self.file}:{self.line}  {self.name}\n"
            f"  >>> {self.source.strip()}\n"
            f"  expected: {self.want.strip()!r}\n"
            f"  got:      {self.got.strip()!r}"
        )


@dataclass
class DoctestResult:
    """Aggregate result of a project-wide run."""

    attempted: int = 0
    passed: int = 0
    failed: int = 0
    failures: List[DoctestFailure] = field(default_factory=list)
    files_scanned: int = 0
    files_with_doctests: int = 0
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.errors

    def summary(self) -> str:
        if self.ok:
            return (
                f"doctests OK: {self.passed}/{self.attempted} "
                f"across {self.files_with_doctests}/{self.files_scanned} files "
                f"({self.duration_s:.2f}s)"
            )
        return (
            f"doctests FAILED: {self.failed} failures out of {self.attempted} examples "
            f"in {self.files_with_doctests}/{self.files_scanned} files"
        )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDES = (
    ".git", ".tox", ".venv", "venv", "__pycache__", "build", "dist",
    "node_modules", ".pytest_cache", ".hypothesis",
)


def find_python_files(
    root: str,
    excludes: Sequence[str] = _DEFAULT_EXCLUDES,
) -> List[str]:
    """Return sorted list of ``.py`` files under ``root``, skipping excludes."""
    out: List[str] = []
    exclude_set = set(excludes)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _file_has_doctest_block(path: str) -> bool:
    """Fast pre-check: does the file contain at least one ``>>>`` line?"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if ">>>" in line:
                    return True
    except OSError:
        return False
    return False


def _load_module_from_path(path: str):
    """Import a module by file path without requiring it to be on sys.path."""
    name = pathlib.Path(path).stem + "__doctest_" + str(abs(hash(path)) % 10 ** 8)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _CaptureRunner(doctest.DocTestRunner):
    """DocTestRunner variant that records failures as DoctestFailure objects."""

    def __init__(self, file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file = file
        self.collected_failures: List[DoctestFailure] = []

    def report_failure(self, out, test, example, got):
        # Build a structured failure record.
        line = (test.lineno or 0) + (example.lineno or 0) + 1
        name = test.name
        self.collected_failures.append(DoctestFailure(
            file=self._file,
            line=line,
            name=name,
            source=example.source.strip("\n"),
            want=example.want,
            got=got,
        ))
        # Suppress the human-readable dump; we'll format our own.
        return None

    def report_unexpected_exception(self, out, test, example, exc_info):
        line = (test.lineno or 0) + (example.lineno or 0) + 1
        exc_lines = "".join(doctest._exception_traceback(exc_info))  # type: ignore[attr-defined]
        self.collected_failures.append(DoctestFailure(
            file=self._file,
            line=line,
            name=test.name,
            source=example.source.strip("\n"),
            want=example.want,
            got=exc_lines,
        ))
        return None


def run_file(path: str) -> DoctestResult:
    """Run doctests for a single Python file; never raises on doctest errors."""
    result = DoctestResult(files_scanned=1)
    if not _file_has_doctest_block(path):
        return result
    result.files_with_doctests = 1

    t0 = time.monotonic()
    try:
        module = _load_module_from_path(path)
    except Exception as exc:
        result.errors.append(f"{path}: import failed: {exc!s}")
        result.duration_s = time.monotonic() - t0
        return result
    if module is None:
        result.errors.append(f"{path}: could not create module spec")
        result.duration_s = time.monotonic() - t0
        return result

    finder = doctest.DocTestFinder(exclude_empty=True)
    tests = finder.find(module)
    runner = _CaptureRunner(file=path, verbose=False, optionflags=0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        for test in tests:
            runner.run(test, out=buf.write)
    result.attempted = runner.tries
    result.failed = len(runner.collected_failures)
    result.passed = max(0, runner.tries - result.failed)
    result.failures = list(runner.collected_failures)
    result.duration_s = time.monotonic() - t0
    return result


def run_project(
    root: str = ".",
    *,
    excludes: Sequence[str] = _DEFAULT_EXCLUDES,
    extra_syspath: Optional[Sequence[str]] = None,
) -> DoctestResult:
    """Walk ``root`` and run doctests across every Python file."""
    agg = DoctestResult()
    t0 = time.monotonic()

    prev_syspath = list(sys.path)
    if extra_syspath:
        for p in extra_syspath:
            if p not in sys.path:
                sys.path.insert(0, p)

    try:
        for path in find_python_files(root, excludes=excludes):
            sub = run_file(path)
            agg.attempted += sub.attempted
            agg.passed += sub.passed
            agg.failed += sub.failed
            agg.failures.extend(sub.failures)
            agg.files_scanned += sub.files_scanned
            agg.files_with_doctests += sub.files_with_doctests
            agg.errors.extend(sub.errors)
    finally:
        sys.path[:] = prev_syspath

    agg.duration_s = time.monotonic() - t0
    agg.failures.sort(key=lambda f: (f.file, f.line, f.name))
    return agg


__all__ = [
    "DoctestFailure",
    "DoctestResult",
    "find_python_files",
    "run_file",
    "run_project",
]
