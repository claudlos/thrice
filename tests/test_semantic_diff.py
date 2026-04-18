"""Tests for ``semantic_diff``."""
from __future__ import annotations

import os
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from semantic_diff import (  # noqa: E402
    diff_python_files,
    diff_python_source,
    diff_python_trees,
)

# ---------------------------------------------------------------------------
# Whitespace / formatting changes do NOT register
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_identical_source_no_changes(self):
        s = "def f():\n    return 1\n"
        r = diff_python_source(s, s)
        assert r.ok and r.changes == []

    def test_whitespace_only(self):
        old = "def f():\n    return 1\n"
        new = "def f():\n    return 1   \n\n"    # extra trailing spaces & blank
        r = diff_python_source(old, new)
        assert r.changes == []

    def test_comment_only_change(self):
        old = "def f():\n    return 1\n"
        new = "def f():\n    # explain\n    return 1\n"
        r = diff_python_source(old, new)
        # Comments are ignored by ast, so body_hash is same.
        assert r.changes == []


# ---------------------------------------------------------------------------
# Structural changes
# ---------------------------------------------------------------------------

class TestAddedRemoved:
    def test_new_function(self):
        old = "def a(): pass\n"
        new = "def a(): pass\ndef b(): pass\n"
        r = diff_python_source(old, new)
        kinds = [c.kind for c in r.changes]
        assert "added" in kinds
        added = [c for c in r.changes if c.kind == "added"][0]
        assert added.qualname == "b"

    def test_removed_function(self):
        old = "def a(): pass\ndef b(): pass\n"
        new = "def a(): pass\n"
        r = diff_python_source(old, new)
        removed = [c for c in r.changes if c.kind == "removed"]
        assert removed and removed[0].qualname == "b"


class TestSignatureChange:
    def test_arg_added(self):
        old = "def f(x): return x\n"
        new = "def f(x, y=0): return x + y\n"
        r = diff_python_source(old, new)
        kinds = {c.kind for c in r.changes}
        assert "signature_changed" in kinds
        assert "body_changed" in kinds   # body hash differs too


class TestBodyChange:
    def test_body_change_only(self):
        old = "def f(x):\n    return x\n"
        new = "def f(x):\n    return x + 1\n"
        r = diff_python_source(old, new)
        kinds = {c.kind for c in r.changes}
        assert "body_changed" in kinds
        assert "signature_changed" not in kinds


class TestDecoratorChange:
    def test_decorator_added(self):
        old = "def f(): pass\n"
        new = "import functools\n@functools.cache\ndef f(): pass\n"
        r = diff_python_source(old, new)
        kinds = {c.kind for c in r.changes}
        assert "decorator_changed" in kinds


class TestRename:
    def test_function_rename_detected(self):
        old = "def foo(x):\n    return x * 2\n"
        new = "def bar(x):\n    return x * 2\n"
        r = diff_python_source(old, new)
        kinds = [c.kind for c in r.changes]
        assert "renamed" in kinds
        assert "added" not in kinds
        assert "removed" not in kinds


# ---------------------------------------------------------------------------
# Classes + methods
# ---------------------------------------------------------------------------

class TestClasses:
    def test_class_method_change_reports_qualname(self):
        old = textwrap.dedent("""\
            class C:
                def m(self):
                    return 1
        """)
        new = textwrap.dedent("""\
            class C:
                def m(self):
                    return 2
        """)
        r = diff_python_source(old, new)
        qns = {c.qualname for c in r.changes}
        assert "C.m" in qns

    def test_new_class_is_added(self):
        old = "\n"
        new = "class C:\n    pass\n"
        r = diff_python_source(old, new)
        kinds = [c.kind for c in r.changes]
        assert "added" in kinds


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_added_import(self):
        old = "def f(): pass\n"
        new = "import os\ndef f(): pass\n"
        r = diff_python_source(old, new)
        kinds = [c.kind for c in r.changes]
        assert "added" in kinds

    def test_reordered_imports_is_not_a_change(self):
        old = "import os\nimport sys\n"
        # Re-order only; same set of imports.
        new = "import sys\nimport os\n"
        r = diff_python_source(old, new)
        assert r.changes == []


# ---------------------------------------------------------------------------
# Fallback on syntax error
# ---------------------------------------------------------------------------

class TestFallback:
    def test_syntax_error_returns_unified_diff(self):
        r = diff_python_source("def a(): pass\n", "def a(:\n    pass\n")
        assert not r.ok
        assert r.fallback_unified and "def a(" in r.fallback_unified


# ---------------------------------------------------------------------------
# File / tree adapters
# ---------------------------------------------------------------------------

class TestFileAdapters:
    def test_diff_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def f(): return 1\n")
        (tmp_path / "b.py").write_text("def f(): return 2\n")
        r = diff_python_files(str(tmp_path / "a.py"), str(tmp_path / "b.py"))
        assert any(c.kind == "body_changed" for c in r.changes)

    def test_diff_trees_detects_file_level_changes(self, tmp_path):
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir(); new.mkdir()
        (old / "x.py").write_text("def f(): return 1\n")
        (new / "x.py").write_text("def f(): return 2\n")
        (new / "y.py").write_text("def g(): return 0\n")
        out = diff_python_trees(str(old), str(new))
        assert set(out) == {"x.py", "y.py"}
        assert any(c.kind == "body_changed" for c in out["x.py"].changes)
        assert any(c.kind == "added" for c in out["y.py"].changes)


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_mentions_bucket_counts(self):
        old = "def a(): return 1\n"
        new = "def a(): return 2\ndef b(): return 3\n"
        r = diff_python_source(old, new)
        s = r.summary()
        assert "added" in s and "body_changed" in s
