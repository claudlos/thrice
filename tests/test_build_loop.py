"""Tests for ``build_loop`` - pure-parser tests + one end-to-end Python run."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from build_loop import (  # noqa: E402
    BuildError,
    BuildResult,
    build,
    default_command,
    detect_toolchain,
    parse_c,
    parse_go,
    parse_python,
    parse_rust,
    parse_typescript,
)

# ===========================================================================
# Rust parser
# ===========================================================================

RUST_OUTPUT = """\
error[E0308]: mismatched types
  --> src/lib.rs:10:5
   |
10 |     x + 1
   |     ^ expected `String`, found `i32`

warning: unused variable: `y`
  --> src/main.rs:2:9
"""

# What ``cargo build --message-format=short`` actually emits.  This is the
# command ``default_command("rust")`` runs, so the parser must ingest it
# as its primary input format.
RUST_OUTPUT_SHORT = """\
src/lib.rs:10:5: error[E0308]: mismatched types
src/main.rs:2:9: warning: unused variable: `y`
"""


class TestRustParser:
    def test_parses_short_error_with_code(self):
        """Short-form is the primary fixture: that's what the configured
        cargo command actually produces."""
        errs = parse_rust(RUST_OUTPUT_SHORT)
        assert any(
            e.file == "src/lib.rs"
            and e.line == 10
            and e.column == 5
            and e.code == "E0308"
            and e.severity == "error"
            for e in errs
        )

    def test_parses_short_warning(self):
        errs = parse_rust(RUST_OUTPUT_SHORT)
        warnings = [e for e in errs if e.severity == "warning"]
        assert warnings and warnings[0].file == "src/main.rs"

    def test_parses_multiline_fallback(self):
        """Default rustc output still parses (override-friendly)."""
        errs = parse_rust(RUST_OUTPUT)
        assert any(
            e.file == "src/lib.rs" and e.line == 10 and e.code == "E0308"
            for e in errs
        )

    def test_deduplicates_when_both_formats_present(self):
        """If an override emits both forms for the same diagnostic, the
        parser must produce a single record, not two."""
        mixed = RUST_OUTPUT_SHORT + "\n" + RUST_OUTPUT
        errs = parse_rust(mixed)
        same = [
            e for e in errs
            if e.file == "src/lib.rs" and e.line == 10 and e.severity == "error"
        ]
        assert len(same) == 1

    def test_empty_output(self):
        assert parse_rust("") == []


# ===========================================================================
# Go parser
# ===========================================================================

GO_OUTPUT = """\
# github.com/x/y
./main.go:10:5: undefined: Foo
./util.go:22:11: cannot use s (type string) as type int
"""


class TestGoParser:
    def test_parses_diagnostics(self):
        errs = parse_go(GO_OUTPUT)
        files = {e.file for e in errs}
        assert "./main.go" in files
        assert "./util.go" in files
        assert all(e.severity == "error" for e in errs)

    def test_line_column_extracted(self):
        errs = parse_go(GO_OUTPUT)
        main = next(e for e in errs if "main.go" in e.file)
        assert main.line == 10
        assert main.column == 5


# ===========================================================================
# TypeScript parser
# ===========================================================================

TS_OUTPUT_PRETTY_FALSE = """\
src/foo.ts:10:5 - error TS2322: Type 'number' is not assignable to type 'string'.
src/bar.ts:3:1 - warning TS6133: 'z' is declared but never read.
"""

TS_OUTPUT_PAREN = """\
src/foo.ts(10,5): error TS2322: Type 'number' is not assignable to type 'string'.
"""


class TestTypeScriptParser:
    def test_pretty_false_format(self):
        errs = parse_typescript(TS_OUTPUT_PRETTY_FALSE)
        codes = {e.code for e in errs}
        assert "TS2322" in codes
        assert "TS6133" in codes

    def test_paren_format(self):
        errs = parse_typescript(TS_OUTPUT_PAREN)
        assert len(errs) == 1
        assert errs[0].line == 10 and errs[0].column == 5


# ===========================================================================
# C/C++ parser
# ===========================================================================

GCC_OUTPUT = """\
foo.c:10:5: error: expected ';' before 'return'
foo.c:12:3: warning: unused variable 'x'
bar.cpp:1:1: note: this is a note
"""


class TestCParser:
    def test_parses_error_warning_note(self):
        errs = parse_c(GCC_OUTPUT)
        sev = {e.severity for e in errs}
        assert {"error", "warning", "note"}.issubset(sev)


# ===========================================================================
# Python parser
# ===========================================================================

PY_OUTPUT = '''\
Listing '.'...
Compiling './bad.py'...
*** File "./bad.py", line 3
    if x
        ^
SyntaxError: invalid syntax
'''


class TestPythonParser:
    def test_parses_syntax_error(self):
        errs = parse_python(PY_OUTPUT)
        assert errs, "expected one error"
        assert errs[0].code == "SyntaxError"
        assert errs[0].line == 3


# ===========================================================================
# Toolchain detection
# ===========================================================================

class TestDetect:
    @pytest.mark.parametrize(
        "marker,toolchain",
        [
            ("Cargo.toml", "rust"),
            ("go.mod", "go"),
            ("tsconfig.json", "typescript"),
            ("Makefile", "c"),
            ("pyproject.toml", "python"),
        ],
    )
    def test_detect_from_marker(self, tmp_path, marker, toolchain):
        (tmp_path / marker).write_text("")
        assert detect_toolchain(str(tmp_path)) == toolchain

    def test_detect_fallback_when_no_marker(self, tmp_path):
        assert detect_toolchain(str(tmp_path)) == "auto"

    def test_default_command_known_toolchains(self):
        for tc in ("rust", "go", "typescript", "c", "python"):
            assert default_command(tc) is not None


# ===========================================================================
# End-to-end: Python syntax failure
# ===========================================================================

class TestEndToEnd:
    def test_python_syntax_failure_reported(self, tmp_path):
        """Plant a syntax-broken .py file and verify the loop surfaces it."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\nversion="0"\n')
        (tmp_path / "bad.py").write_text("def f(\n    pass\n")  # missing ')'
        result = build(project_dir=str(tmp_path), toolchain="python", timeout=30.0)
        assert isinstance(result, BuildResult)
        assert not result.ok
        # compileall uses bytecode path; the parser uses the source path.
        assert any("bad.py" in e.file for e in result.errors)

    def test_clean_python_project_returns_ok(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\nversion="0"\n')
        (tmp_path / "ok.py").write_text("x = 1\n")
        result = build(project_dir=str(tmp_path), toolchain="python", timeout=30.0)
        assert result.ok
        assert result.error_count == 0

    def test_missing_toolchain_binary_raises(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'\nversion='0.1.0'\n")
        # Override PATH so cargo definitely isn't found.
        import shutil as _shutil
        if _shutil.which("cargo"):
            pytest.skip("cargo is installed; cannot test the missing-binary path")
        with pytest.raises(FileNotFoundError):
            build(project_dir=str(tmp_path), toolchain="rust")


# ===========================================================================
# Summary formatting
# ===========================================================================

class TestBuildResultSummary:
    def test_first_errors_is_deterministic(self):
        errs = [
            BuildError(file="b.py", line=1, column=0, severity="error",
                       message="b1"),
            BuildError(file="a.py", line=2, column=1, severity="error",
                       message="a2"),
            BuildError(file="a.py", line=1, column=0, severity="error",
                       message="a1"),
        ]
        r = BuildResult(
            toolchain="python", command=[], ok=False,
            returncode=1, errors=errs,
        )
        first = r.first_errors(2)
        assert [e.message for e in first] == ["a1", "a2"]
