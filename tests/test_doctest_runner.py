"""Tests for ``doctest_runner``."""
from __future__ import annotations

import os
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from doctest_runner import (  # noqa: E402
    find_python_files,
    run_file,
    run_project,
)

# ---------------------------------------------------------------------------
# find_python_files
# ---------------------------------------------------------------------------

class TestFind:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert len(files) == 2
        assert any(p.endswith("a.py") for p in files)
        assert any(p.endswith("b.py") for p in files)

    def test_excludes_dunder_dirs(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "c.py").write_text("")
        (tmp_path / "keep.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert all("__pycache__" not in p for p in files)
        assert any(p.endswith("keep.py") for p in files)


# ---------------------------------------------------------------------------
# run_file
# ---------------------------------------------------------------------------

_PASSING = textwrap.dedent('''\
    """Module with passing doctests."""

    def add(a, b):
        """Add two numbers.

        >>> add(2, 3)
        5
        >>> add(-1, 1)
        0
        """
        return a + b
''')

_FAILING = textwrap.dedent('''\
    """Module with one failing doctest."""

    def identity(x):
        """Return x.

        >>> identity(1)
        2
        """
        return x
''')

_RAISING = textwrap.dedent('''\
    """Module whose doctest raises."""

    def bad():
        """
        >>> bad()
        'ok'
        """
        raise RuntimeError("boom")
''')

_NO_DOCTESTS = textwrap.dedent('''\
    """No doctests here at all."""

    def f():
        return 1
''')


class TestRunFile:
    def test_passing(self, tmp_path):
        p = tmp_path / "pass.py"
        p.write_text(_PASSING)
        r = run_file(str(p))
        assert r.ok
        assert r.attempted == 2
        assert r.passed == 2
        assert r.failed == 0
        assert not r.failures

    def test_failing(self, tmp_path):
        p = tmp_path / "fail.py"
        p.write_text(_FAILING)
        r = run_file(str(p))
        assert not r.ok
        assert r.failed == 1
        assert r.failures and r.failures[0].want.strip() == "2"

    def test_raising(self, tmp_path):
        p = tmp_path / "raise.py"
        p.write_text(_RAISING)
        r = run_file(str(p))
        assert r.failed == 1
        assert "RuntimeError" in r.failures[0].got

    def test_no_doctests(self, tmp_path):
        p = tmp_path / "plain.py"
        p.write_text(_NO_DOCTESTS)
        r = run_file(str(p))
        assert r.attempted == 0
        assert r.passed == 0
        assert r.files_with_doctests == 0

    def test_import_error_captured(self, tmp_path):
        # Missing closing paren => module won't even import.
        p = tmp_path / "bad_syntax.py"
        p.write_text('"""\n>>> x\n1\n"""\ndef f(:\n    pass\n')
        r = run_file(str(p))
        assert r.errors
        assert any("import failed" in e for e in r.errors)


# ---------------------------------------------------------------------------
# run_project
# ---------------------------------------------------------------------------

class TestRunProject:
    def test_aggregates_passes_and_failures(self, tmp_path):
        (tmp_path / "good.py").write_text(_PASSING)
        (tmp_path / "bad.py").write_text(_FAILING)
        r = run_project(str(tmp_path))
        assert r.attempted == 3        # 2 from good + 1 from bad
        assert r.failed == 1
        assert r.passed == 2
        assert r.files_with_doctests == 2
        assert r.files_scanned == 2

    def test_summary_readable(self, tmp_path):
        (tmp_path / "good.py").write_text(_PASSING)
        r = run_project(str(tmp_path))
        assert "OK" in r.summary() and "2/2" in r.summary()

    def test_failure_sort_is_deterministic(self, tmp_path):
        (tmp_path / "b.py").write_text(_FAILING)
        (tmp_path / "a.py").write_text(_FAILING)
        r = run_project(str(tmp_path))
        # Two failures, sorted by file.
        files = [f.file for f in r.failures]
        assert files == sorted(files)

    def test_extra_syspath_honored(self, tmp_path):
        """Callers can pass extra directories to go on sys.path."""
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "mydep.py").write_text("VALUE = 42\n")

        caller = tmp_path / "caller.py"
        caller.write_text(textwrap.dedent('''\
            """Needs mydep on sys.path.

            >>> from mydep import VALUE
            >>> VALUE
            42
            """
        '''))

        # Without extra_syspath the import would fail:
        r_bad = run_project(str(tmp_path),
                            excludes=("lib", ".git", "__pycache__"))
        # Could be failure OR error depending on doctest discovery.
        assert not r_bad.ok

        r_ok = run_project(
            str(tmp_path),
            excludes=("lib", ".git", "__pycache__"),
            extra_syspath=[str(lib)],
        )
        assert r_ok.ok, (r_ok.summary(), r_ok.errors, r_ok.failures)
