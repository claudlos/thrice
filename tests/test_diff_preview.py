"""Tests for ``diff_preview``."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from diff_preview import (  # noqa: E402
    EditVerdict,
    preview_content,
    preview_edit,
    preview_patch,
)

# ---------------------------------------------------------------------------
# Python adapter
# ---------------------------------------------------------------------------

class TestPython:
    def test_valid_python_passes(self):
        v = preview_content("foo.py", "def add(a, b):\n    return a + b\n")
        assert v.ok and v.language == "python" and v.issues == []

    def test_broken_python_blocked(self):
        v = preview_content("foo.py", "def add(\n    return 1\n")
        assert not v.ok
        # Should carry a Python-specific issue.
        assert any(i.code in ("SYNTAX_ERROR", "UNBALANCED_BRACKETS") for i in v.issues)

    def test_indent_error_caught(self):
        v = preview_content("foo.py", "def f():\n  x = 1\n    y = 2\n")
        assert not v.ok

    def test_docstring_with_braces_is_fine(self):
        # Triple-quoted strings shouldn't upset bracket balance.
        src = '''def f():
    """Example: {a: 1, b: 2}"""
    return 0
'''
        v = preview_content("foo.py", src)
        assert v.ok, v.summary()


# ---------------------------------------------------------------------------
# JSON adapter
# ---------------------------------------------------------------------------

class TestJSON:
    def test_valid_json(self):
        v = preview_content("c.json", '{"a": 1, "b": [1, 2]}')
        assert v.ok

    def test_invalid_json(self):
        v = preview_content("c.json", "{bad: json,}")
        assert not v.ok
        assert any(i.code == "JSON_PARSE_ERROR" for i in v.issues)

    def test_empty_json_blocked(self):
        v = preview_content("c.json", "")
        assert not v.ok
        assert any(i.code == "EMPTY_FILE" for i in v.issues)


# ---------------------------------------------------------------------------
# TOML adapter (3.11+)
# ---------------------------------------------------------------------------

class TestTOML:
    def test_valid_toml(self):
        v = preview_content("a.toml", '[x]\nkey = "value"\n')
        # On 3.10 tomllib is absent and we pass-through; on 3.11+ this must pass.
        assert v.ok

    def test_invalid_toml(self):
        pytest.importorskip("tomllib")
        v = preview_content("a.toml", "this is = = not toml\n")
        assert not v.ok


# ---------------------------------------------------------------------------
# Universal layer
# ---------------------------------------------------------------------------

class TestUniversal:
    def test_conflict_marker_blocks_any_language(self):
        src = "\n".join([
            "print('hi')",
            "<<<<<<< HEAD",
            "a = 1",
            ">>>>>>> branch",
            "",
        ])
        v = preview_content("foo.py", src)
        assert not v.ok
        assert any(i.code == "CONFLICT_MARKER" for i in v.issues)

    def test_unbalanced_brackets_generic_file(self):
        # Unknown extension => generic checks only.
        src = "{ 'x': ["
        v = preview_content("foo.unknown", src)
        assert not v.ok
        assert any(i.code == "UNBALANCED_BRACKETS" for i in v.issues)

    def test_generic_balanced_is_ok(self):
        v = preview_content("foo.txt", "just text.\nNo brackets here.\n")
        assert v.ok


# ---------------------------------------------------------------------------
# preview_patch
# ---------------------------------------------------------------------------

_ORIG = "line one\nline two\nline three\n"

_PATCH_OK = """\
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 line one
-line two
+line TWO
 line three
"""

_PATCH_BROKEN = """\
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 line ZERO
-line two
+line TWO
 line three
"""


class TestPreviewPatch:
    def test_clean_patch_yields_parseable_text(self):
        # Replace the content with a valid Python module.
        orig = "x = 1\ny = 2\nz = 3\n"
        patch = """\
--- a/m.py
+++ b/m.py
@@ -1,3 +1,3 @@
 x = 1
-y = 2
+y = 22
 z = 3
"""
        v = preview_patch("m.py", orig, patch)
        assert v.ok, v.summary()

    def test_patch_apply_failure_reported(self):
        v = preview_patch("x.py", _ORIG, _PATCH_BROKEN)
        assert not v.ok
        assert any(i.code == "PATCH_APPLY_FAILED" for i in v.issues)


# ---------------------------------------------------------------------------
# Integration: matches expected shape
# ---------------------------------------------------------------------------

class TestAPI:
    def test_preview_edit_alias(self):
        v = preview_edit("a.py", "x = 1\n")
        assert isinstance(v, EditVerdict) and v.ok

    def test_summary_text_nonempty(self):
        v = preview_edit("a.py", "def (\n    pass\n")
        s = v.summary()
        assert "BLOCKED" in s or "OK" in s
