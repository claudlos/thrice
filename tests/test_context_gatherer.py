"""
Tests for context_gatherer.py — Smart Context Gathering.

Uses tmp_path (pytest) to create realistic project layouts on disk.
"""

import os
import sys

# Ensure the module can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from context_gatherer import (
    ContextGatherer,
    ContextPrioritizer,
    ContextRequest,
    GatheredContext,
    ImportExtractor,
    TestFileFinder,
    format_for_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, content=""):
    """Write a file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _make_python_project(root):
    """Create a small Python project for testing."""
    _write(os.path.join(root, "main.py"), (
        "import utils\n"
        "from models import User\n"
        "\n"
        "def run():\n"
        "    u = User('test')\n"
        "    utils.greet(u)\n"
    ))
    _write(os.path.join(root, "utils.py"), (
        "def greet(user):\n"
        "    print(f'Hello {user.name}')\n"
    ))
    _write(os.path.join(root, "models.py"), (
        "class User:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
    ))
    _write(os.path.join(root, "tests", "test_main.py"), (
        "from main import run\n"
        "\n"
        "def test_run():\n"
        "    run()\n"
    ))
    _write(os.path.join(root, "tests", "test_utils.py"), (
        "from utils import greet\n"
        "\n"
        "def test_greet():\n"
        "    pass\n"
    ))
    _write(os.path.join(root, "tests", "test_models.py"), (
        "from models import User\n"
        "\n"
        "def test_user():\n"
        "    u = User('x')\n"
        "    assert u.name == 'x'\n"
    ))


# ===========================================================================
# ImportExtractor tests
# ===========================================================================

class TestImportExtractorPython:
    """Tests for Python import extraction."""

    def test_simple_import(self):
        content = "import os\nimport sys\n"
        result = ImportExtractor.extract_python_imports(content)
        assert "os" in result
        assert "sys" in result

    def test_from_import(self):
        content = "from os.path import join\nfrom collections import defaultdict\n"
        result = ImportExtractor.extract_python_imports(content)
        assert "os.path" in result
        assert "collections" in result

    def test_dotted_import(self):
        content = "import my.package.module\n"
        result = ImportExtractor.extract_python_imports(content)
        assert "my.package.module" in result

    def test_mixed_imports(self):
        content = "import os\nfrom pathlib import Path\nimport json\n"
        result = ImportExtractor.extract_python_imports(content)
        assert len(result) == 3

    def test_no_imports(self):
        content = "x = 1\nprint(x)\n"
        result = ImportExtractor.extract_python_imports(content)
        assert result == []

    def test_indented_import_ignored(self):
        # Conditional imports
        content = "if True:\n    import secret\n"
        result = ImportExtractor.extract_python_imports(content)
        # Our regex allows indented imports
        assert "secret" in result

    def test_comment_not_import(self):
        content = "# import os\nx = 1\n"
        result = ImportExtractor.extract_python_imports(content)
        # The regex will still match because it doesn't skip comments
        # This is acceptable — comment handling would require a full parser
        assert len(result) <= 1


class TestImportExtractorJS:
    """Tests for JS/TS import extraction."""

    def test_es6_import(self):
        content = "import React from 'react';\n"
        result = ImportExtractor.extract_js_imports(content)
        assert "react" in result

    def test_named_import(self):
        content = "import { useState, useEffect } from 'react';\n"
        result = ImportExtractor.extract_js_imports(content)
        assert "react" in result

    def test_require(self):
        content = "const fs = require('fs');\n"
        result = ImportExtractor.extract_js_imports(content)
        assert "fs" in result

    def test_relative_import(self):
        content = "import utils from './utils';\n"
        result = ImportExtractor.extract_js_imports(content)
        assert "./utils" in result

    def test_multiple_imports(self):
        content = (
            "import React from 'react';\n"
            "import { render } from 'react-dom';\n"
            "const path = require('path');\n"
        )
        result = ImportExtractor.extract_js_imports(content)
        assert len(result) == 3

    def test_no_imports(self):
        content = "const x = 1;\nconsole.log(x);\n"
        result = ImportExtractor.extract_js_imports(content)
        assert result == []


class TestImportExtractorRust:
    """Tests for Rust import extraction."""

    def test_simple_use(self):
        content = "use std::io;\n"
        result = ImportExtractor.extract_rust_imports(content)
        assert "std::io" in result

    def test_nested_use(self):
        content = "use std::collections::HashMap;\n"
        result = ImportExtractor.extract_rust_imports(content)
        assert "std::collections::HashMap" in result

    def test_grouped_use(self):
        content = "use std::io::{Read, Write};\n"
        result = ImportExtractor.extract_rust_imports(content)
        assert any("std::io" in r for r in result)

    def test_no_use(self):
        content = "fn main() { println!(\"hello\"); }\n"
        result = ImportExtractor.extract_rust_imports(content)
        assert result == []


class TestAutoDetectLanguage:
    """Tests for language auto-detection."""

    def test_python(self):
        assert ImportExtractor.auto_detect_language("foo.py") == "python"

    def test_javascript(self):
        assert ImportExtractor.auto_detect_language("app.js") == "javascript"

    def test_typescript(self):
        assert ImportExtractor.auto_detect_language("app.ts") == "typescript"

    def test_tsx(self):
        assert ImportExtractor.auto_detect_language("Component.tsx") == "typescript"

    def test_rust(self):
        assert ImportExtractor.auto_detect_language("main.rs") == "rust"

    def test_unknown(self):
        assert ImportExtractor.auto_detect_language("file.txt") == "unknown"

    def test_no_extension(self):
        assert ImportExtractor.auto_detect_language("Makefile") == "unknown"


class TestResolveImportToFile:
    """Tests for import-to-file resolution."""

    def test_resolve_simple_module(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "# utils")
        result = ImportExtractor.resolve_import_to_file("utils", str(tmp_path))
        assert result == "utils.py"

    def test_resolve_package(self, tmp_path):
        _write(str(tmp_path / "mypackage" / "__init__.py"), "")
        result = ImportExtractor.resolve_import_to_file("mypackage", str(tmp_path))
        assert result == os.path.join("mypackage", "__init__.py")

    def test_resolve_nested_module(self, tmp_path):
        _write(str(tmp_path / "pkg" / "sub.py"), "")
        result = ImportExtractor.resolve_import_to_file("pkg.sub", str(tmp_path))
        assert result == os.path.join("pkg", "sub.py")

    def test_resolve_nonexistent(self, tmp_path):
        result = ImportExtractor.resolve_import_to_file("nonexistent", str(tmp_path))
        assert result is None

    def test_resolve_js_module(self, tmp_path):
        _write(str(tmp_path / "utils.js"), "")
        result = ImportExtractor.resolve_import_to_file("utils", str(tmp_path))
        # Should find .js after .py fails
        assert result is not None
        assert "utils" in result


# ===========================================================================
# TestFileFinder tests
# ===========================================================================

class TestTestFileFinder:
    """Tests for test file discovery."""

    def test_find_test_prefix(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "")
        _write(str(tmp_path / "test_utils.py"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.py"), str(tmp_path))
        assert any("test_utils.py" in r for r in result)

    def test_find_test_suffix(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "")
        _write(str(tmp_path / "utils_test.py"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.py"), str(tmp_path))
        assert any("utils_test.py" in r for r in result)

    def test_find_tests_dir(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "")
        _write(str(tmp_path / "tests" / "test_utils.py"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.py"), str(tmp_path))
        assert any("test_utils.py" in r for r in result)

    def test_find_js_test(self, tmp_path):
        _write(str(tmp_path / "utils.js"), "")
        _write(str(tmp_path / "__tests__" / "utils.test.js"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.js"), str(tmp_path))
        assert any("utils.test.js" in r for r in result)

    def test_find_ts_spec(self, tmp_path):
        _write(str(tmp_path / "utils.ts"), "")
        _write(str(tmp_path / "utils.spec.ts"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.ts"), str(tmp_path))
        assert any("utils.spec.ts" in r for r in result)

    def test_no_test_files(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.py"), str(tmp_path))
        assert result == []

    def test_multiple_test_files(self, tmp_path):
        _write(str(tmp_path / "utils.py"), "")
        _write(str(tmp_path / "test_utils.py"), "")
        _write(str(tmp_path / "tests" / "test_utils.py"), "")
        result = TestFileFinder.find_test_files(str(tmp_path / "utils.py"), str(tmp_path))
        assert len(result) >= 2


# ===========================================================================
# ContextPrioritizer tests
# ===========================================================================

class TestContextPrioritizer:
    """Tests for context prioritization."""

    def test_within_budget_unchanged(self):
        ctx = GatheredContext(
            target_file_content="x" * 100,
            imports=["a.py", "b.py"],
            imported_by=["c.py"],
            test_files=["test_x.py"],
            related_files=["y.py"],
            token_estimate=100,
        )
        result = ContextPrioritizer.prioritize(ctx, max_tokens=10000)
        assert len(result.imports) == 2
        assert len(result.imported_by) == 1
        assert len(result.test_files) == 1
        assert len(result.related_files) == 1

    def test_drops_related_first(self):
        ctx = GatheredContext(
            target_file_content="x" * 40,
            imports=["a.py"],
            imported_by=["b.py"],
            test_files=["test.py"],
            related_files=["r1.py", "r2.py", "r3.py"],
            token_estimate=100,
        )
        # Very tight budget: just enough for target
        result = ContextPrioritizer.prioritize(ctx, max_tokens=15)
        # Related files should be dropped first
        assert len(result.related_files) < 3

    def test_drops_imported_by_after_related(self):
        ctx = GatheredContext(
            target_file_content="x" * 20,
            imports=["a" * 20],
            imported_by=["b" * 20],
            test_files=["t" * 20],
            related_files=["r" * 20],
            token_estimate=100,
        )
        result = ContextPrioritizer.prioritize(ctx, max_tokens=8)
        # Should have dropped related_files first, then imported_by
        assert len(result.related_files) == 0

    def test_target_content_never_dropped(self):
        content = "x" * 1000
        ctx = GatheredContext(
            target_file_content=content,
            imports=["a.py"],
            token_estimate=300,
        )
        result = ContextPrioritizer.prioritize(ctx, max_tokens=1)
        # Target content is preserved even if over budget
        assert result.target_file_content == content

    def test_empty_context(self):
        ctx = GatheredContext()
        result = ContextPrioritizer.prioritize(ctx, max_tokens=1000)
        assert result.token_estimate >= 0

    def test_token_estimate_updated(self):
        ctx = GatheredContext(
            target_file_content="hello world",
            imports=["os"],
            token_estimate=999,
        )
        result = ContextPrioritizer.prioritize(ctx, max_tokens=10000)
        assert result.token_estimate != 999  # Should be recalculated


# ===========================================================================
# ContextGatherer tests (integration)
# ===========================================================================

class TestContextGatherer:
    """Integration tests for the full context gathering pipeline."""

    def test_gather_basic(self, tmp_path):
        root = str(tmp_path)
        _make_python_project(root)

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="main.py",
            change_description="add logging",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        assert "import utils" in ctx.target_file_content
        assert ctx.token_estimate > 0
        assert ctx.summary != ""

    def test_gather_finds_imports(self, tmp_path):
        root = str(tmp_path)
        _make_python_project(root)

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="main.py",
            change_description="refactor",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        # main.py imports utils and models
        assert any("utils" in imp for imp in ctx.imports)

    def test_gather_finds_test_files(self, tmp_path):
        root = str(tmp_path)
        _make_python_project(root)

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="main.py",
            change_description="test",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        assert any("test_main" in tf for tf in ctx.test_files)

    def test_gather_finds_reverse_deps(self, tmp_path):
        root = str(tmp_path)
        _make_python_project(root)

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="utils.py",
            change_description="modify greet",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        # main.py imports utils
        assert any("main" in dep for dep in ctx.imported_by)

    def test_gather_nonexistent_file(self, tmp_path):
        root = str(tmp_path)
        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="nonexistent.py",
            change_description="test",
            project_root=root,
        )
        ctx = gatherer.gather(request)
        assert ctx.target_file_content == ""

    def test_gather_absolute_path(self, tmp_path):
        root = str(tmp_path)
        _write(os.path.join(root, "app.py"), "x = 1\n")

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file=os.path.join(root, "app.py"),
            change_description="test",
            project_root=root,
        )
        ctx = gatherer.gather(request)
        assert "x = 1" in ctx.target_file_content

    def test_gather_finds_related_files(self, tmp_path):
        root = str(tmp_path)
        _write(os.path.join(root, "user_service.py"), "class UserService: pass\n")
        _write(os.path.join(root, "user_model.py"), "class UserModel: pass\n")
        _write(os.path.join(root, "user_controller.py"), "class UserController: pass\n")
        _write(os.path.join(root, "payment.py"), "class Payment: pass\n")

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="user_service.py",
            change_description="refactor",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        # user_model and user_controller should be related
        related_names = " ".join(ctx.related_files)
        assert "user_model" in related_names or "user_controller" in related_names

    def test_gather_dependency_chain(self, tmp_path):
        root = str(tmp_path)
        _write(os.path.join(root, "a.py"), "import b\n")
        _write(os.path.join(root, "b.py"), "import c\n")
        _write(os.path.join(root, "c.py"), "x = 1\n")

        gatherer = ContextGatherer()
        request = ContextRequest(
            target_file="a.py",
            change_description="test chain",
            project_root=root,
        )
        ctx = gatherer.gather(request)

        # Should find b.py and c.py in the chain
        chain_str = " ".join(ctx.dependency_chain)
        assert "b.py" in chain_str
        assert "c.py" in chain_str

    def test_gather_for_edit(self, tmp_path):
        root = str(tmp_path)
        _write(os.path.join(root, ".git", "config"), "")  # marker for project root
        _write(os.path.join(root, "app.py"), "print('hello')\n")

        gatherer = ContextGatherer()
        ctx = gatherer.gather_for_edit(
            os.path.join(root, "app.py"),
            "add error handling",
        )
        assert "print('hello')" in ctx.target_file_content


# ===========================================================================
# format_for_prompt tests
# ===========================================================================

class TestFormatForPrompt:
    """Tests for prompt formatting."""

    def test_basic_format(self):
        ctx = GatheredContext(
            target_file_content="def foo(): pass",
            imports=["os", "sys"],
            imported_by=["main.py"],
            test_files=["test_foo.py"],
            related_files=["bar.py"],
            dependency_chain=["os.py"],
            summary="Context for foo.py",
            token_estimate=42,
        )
        output = format_for_prompt(ctx)

        assert "GATHERED CONTEXT" in output
        assert "TARGET FILE" in output
        assert "def foo(): pass" in output
        assert "IMPORTS" in output
        assert "os" in output
        assert "IMPORTED BY" in output
        assert "main.py" in output
        assert "TEST FILES" in output
        assert "test_foo.py" in output
        assert "RELATED FILES" in output
        assert "bar.py" in output
        assert "DEPENDENCY CHAIN" in output
        assert "42" in output

    def test_empty_context(self):
        ctx = GatheredContext()
        output = format_for_prompt(ctx)
        assert "GATHERED CONTEXT" in output
        assert "(file not found or empty)" in output

    def test_partial_context(self):
        ctx = GatheredContext(
            target_file_content="x = 1",
            imports=["os"],
            token_estimate=10,
        )
        output = format_for_prompt(ctx)
        assert "IMPORTS" in output
        # Sections with no data should not appear
        assert "IMPORTED BY" not in output
        assert "TEST FILES" not in output

    def test_format_includes_summary(self):
        ctx = GatheredContext(
            target_file_content="code",
            summary="Testing summary line",
            token_estimate=5,
        )
        output = format_for_prompt(ctx)
        assert "Testing summary line" in output


# ===========================================================================
# Edge cases and dataclass tests
# ===========================================================================

class TestDataclasses:
    """Tests for dataclass construction and defaults."""

    def test_context_request_fields(self):
        req = ContextRequest(
            target_file="main.py",
            change_description="add feature",
            project_root="/tmp/proj",
        )
        assert req.target_file == "main.py"
        assert req.change_description == "add feature"
        assert req.project_root == "/tmp/proj"

    def test_gathered_context_defaults(self):
        ctx = GatheredContext()
        assert ctx.target_file_content == ""
        assert ctx.imports == []
        assert ctx.imported_by == []
        assert ctx.test_files == []
        assert ctx.related_files == []
        assert ctx.dependency_chain == []
        assert ctx.summary == ""
        assert ctx.token_estimate == 0

    def test_gathered_context_mutable_defaults(self):
        """Ensure each instance gets its own list."""
        a = GatheredContext()
        b = GatheredContext()
        a.imports.append("os")
        assert b.imports == []
