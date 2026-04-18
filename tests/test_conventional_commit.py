"""Tests for ``conventional_commit``."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from conventional_commit import (  # noqa: E402
    VALID_TYPES,
    generate,
    suggest_type,
    validate,
)

# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_minimal_happy_path(self):
        r = validate("feat: add thing")
        assert r.ok and r.type == "feat" and r.description == "add thing"

    def test_with_scope(self):
        r = validate("fix(auth): handle empty session")
        assert r.ok and r.scope == "auth"

    def test_breaking_bang(self):
        r = validate("feat(api)!: drop /v1 endpoints")
        assert r.ok and r.breaking

    def test_breaking_footer(self):
        msg = "refactor: rename foo to bar\n\nbody line\n\nBREAKING CHANGE: foo is gone"
        r = validate(msg)
        assert r.ok and r.breaking

    def test_body_preserved(self):
        msg = "feat: thing\n\nFirst paragraph.\n\nSecond paragraph."
        r = validate(msg)
        assert r.ok
        assert r.body and "First paragraph" in r.body
        assert "Second paragraph" in r.body

    def test_footers_detected(self):
        msg = "fix: stuff\n\nbody\n\nRefs #42\nReviewed-by: Alice"
        r = validate(msg)
        assert r.ok
        assert any("Refs #42" in f for f in r.footers)
        assert any("Reviewed-by" in f for f in r.footers)

    def test_empty_rejected(self):
        assert not validate("").ok
        assert not validate("   ").ok

    def test_bad_type(self):
        r = validate("spicy: oops")
        assert not r.ok
        assert any("unknown type" in e for e in r.errors)

    def test_description_period_rejected(self):
        r = validate("feat: add thing.")
        assert not r.ok
        assert any("period" in e for e in r.errors)

    def test_description_uppercase_rejected(self):
        r = validate("feat: Capital")
        assert not r.ok
        assert any("lowercase" in e for e in r.errors)

    def test_header_too_long(self):
        r = validate("feat: " + "x" * 80)
        assert not r.ok
        assert any("too long" in e for e in r.errors)

    def test_missing_blank_line_after_header(self):
        r = validate("feat: thing\nbody without blank")
        assert not r.ok
        assert any("blank" in e for e in r.errors)

    def test_bang_only_on_breakable_types(self):
        # ``docs!`` is not a valid breaking type.
        r = validate("docs!: rewrite readme")
        assert not r.ok
        assert any("breaking" in e.lower() for e in r.errors)

    def test_all_canonical_types_validate(self):
        for t in VALID_TYPES:
            r = validate(f"{t}: do stuff")
            assert r.ok, (t, r.errors)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_roundtrip(self):
        msg = generate("feat", "add thing", scope="cron")
        r = validate(msg)
        assert r.ok and r.scope == "cron" and r.description == "add thing"

    def test_roundtrip_with_body_and_footer(self):
        msg = generate(
            "fix", "handle None",
            scope="auth",
            body="Previously raised NullReference.",
            footers=["Refs #99"],
        )
        r = validate(msg)
        assert r.ok
        assert r.body and "NullReference" in r.body
        assert "Refs #99" in r.footers[0]

    def test_breaking(self):
        msg = generate("feat", "remove v1", breaking=True)
        r = validate(msg)
        assert r.ok and r.breaking


# ---------------------------------------------------------------------------
# suggest_type()
# ---------------------------------------------------------------------------

_DIFF_TESTS_ONLY = """\
diff --git a/tests/test_x.py b/tests/test_x.py
index 0000..1111
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1 +1,2 @@
 x
+y
"""

_DIFF_DOCS_ONLY = """\
diff --git a/README.md b/README.md
index 0000..1111
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 title
+paragraph
"""

_DIFF_CI = """\
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 0000..1111
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -1 +1,2 @@
 steps
+more
"""

_DIFF_MIXED_SRC_AND_TEST = """\
diff --git a/src/foo.py b/src/foo.py
index 0000..1111
--- a/src/foo.py
+++ b/src/foo.py
@@ -1 +1,2 @@
 a
+b
diff --git a/tests/test_foo.py b/tests/test_foo.py
index 0000..1111
--- a/tests/test_foo.py
+++ b/tests/test_foo.py
@@ -1 +1,2 @@
 a
+b
"""

_DIFF_BUILD = """\
diff --git a/pyproject.toml b/pyproject.toml
index 0000..1111
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1 +1,2 @@
 x
+y
"""

_DIFF_NEW_FILE = """\
diff --git a/src/new_feature.py b/src/new_feature.py
new file mode 100644
index 0000..1111
--- /dev/null
+++ b/src/new_feature.py
@@ -0,0 +1 @@
+def hello(): ...
"""


class TestSuggestType:
    @pytest.mark.parametrize(
        "diff,expected",
        [
            (_DIFF_TESTS_ONLY, "test"),
            (_DIFF_DOCS_ONLY, "docs"),
            (_DIFF_CI, "ci"),
            (_DIFF_MIXED_SRC_AND_TEST, "fix"),
            (_DIFF_BUILD, "build"),
            (_DIFF_NEW_FILE, "feat"),
        ],
    )
    def test_suggested_type(self, diff, expected):
        assert suggest_type(diff) == expected

    def test_paths_override(self):
        # Caller can bypass diff parsing and pass paths directly.
        assert suggest_type("", paths=["src/new.py"]) == "refactor"
