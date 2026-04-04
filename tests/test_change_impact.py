"""
Tests for change_impact.py — Dependency-Aware Change Impact Analysis.

30+ tests covering ChangeDetector, ImpactAnalyzer, ImpactReport, and helpers.
"""

import os
import sys
import pytest

# Add the new-files directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'new-files'))

from change_impact import (
    ChangeType,
    DetectedChange,
    Usage,
    ImpactedFile,
    ChangeDetector,
    ImpactAnalyzer,
    ImpactReport,
    quick_impact_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return ChangeDetector()


@pytest.fixture
def analyzer():
    return ImpactAnalyzer()


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal project structure for impact analysis."""
    # Main module that will be "changed"
    main = tmp_path / "main.py"
    main.write_text(
        "def process_data(data, format='json'):\n"
        "    return data\n"
        "\n"
        "class DataProcessor:\n"
        "    def run(self):\n"
        "        pass\n"
    )

    # A file that imports and calls process_data
    caller = tmp_path / "caller.py"
    caller.write_text(
        "from main import process_data, DataProcessor\n"
        "\n"
        "def do_work():\n"
        "    result = process_data(my_data, format='csv')\n"
        "    return result\n"
        "\n"
        "class MyCaller(DataProcessor):\n"
        "    pass\n"
    )

    # A file with a type hint reference
    types_file = tmp_path / "types_usage.py"
    types_file.write_text(
        "from main import DataProcessor\n"
        "\n"
        "def get_processor() -> DataProcessor:\n"
        "    return DataProcessor()\n"
    )

    # A file with string reference
    docs_file = tmp_path / "docs.py"
    docs_file.write_text(
        '# This module documents process_data usage\n'
        'HELP_TEXT = "Use process_data to transform"\n'
    )

    return tmp_path


# ---------------------------------------------------------------------------
# ChangeType enum tests
# ---------------------------------------------------------------------------

class TestChangeType:
    def test_all_change_types_exist(self):
        expected = [
            "FUNCTION_RENAMED", "FUNCTION_SIGNATURE_CHANGED", "FUNCTION_REMOVED",
            "CLASS_RENAMED", "CLASS_REMOVED", "METHOD_CHANGED",
            "PARAMETER_ADDED", "PARAMETER_REMOVED", "PARAMETER_RENAMED",
            "RETURN_TYPE_CHANGED", "IMPORT_CHANGED", "CONSTANT_CHANGED",
        ]
        for name in expected:
            assert hasattr(ChangeType, name), f"Missing ChangeType.{name}"

    def test_change_type_values(self):
        assert ChangeType.FUNCTION_RENAMED.value == "function_renamed"
        assert ChangeType.CLASS_REMOVED.value == "class_removed"


# ---------------------------------------------------------------------------
# DetectedChange dataclass tests
# ---------------------------------------------------------------------------

class TestDetectedChange:
    def test_create_detected_change(self):
        change = DetectedChange(
            change_type=ChangeType.FUNCTION_REMOVED,
            file_path="test.py",
            symbol_name="foo",
            old_signature="def foo(x)",
            new_signature="",
            line_number=10,
        )
        assert change.change_type == ChangeType.FUNCTION_REMOVED
        assert change.file_path == "test.py"
        assert change.symbol_name == "foo"
        assert change.line_number == 10

    def test_defaults(self):
        change = DetectedChange(
            change_type=ChangeType.FUNCTION_REMOVED,
            file_path="test.py",
            symbol_name="foo",
        )
        assert change.old_signature == ""
        assert change.new_signature == ""
        assert change.line_number == 0


# ---------------------------------------------------------------------------
# Usage / ImpactedFile dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_usage_creation(self):
        u = Usage(line_number=5, line_content="foo()", usage_type="call")
        assert u.line_number == 5
        assert u.usage_type == "call"

    def test_impacted_file_defaults(self):
        f = ImpactedFile(file_path="test.py")
        assert f.usages == []
        assert f.risk_level == "low"
        assert f.suggested_update == ""


# ---------------------------------------------------------------------------
# ChangeDetector tests
# ---------------------------------------------------------------------------

class TestChangeDetector:
    def test_detect_function_removed(self, detector):
        old = "def foo(x):\n    return x\n"
        new = "# foo was removed\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.FUNCTION_REMOVED for c in changes)
        removed = [c for c in changes if c.change_type == ChangeType.FUNCTION_REMOVED]
        assert removed[0].symbol_name == "foo"

    def test_detect_function_signature_changed(self, detector):
        old = "def foo(x):\n    return x\n"
        new = "def foo(x, y):\n    return x + y\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert len(changes) >= 1
        assert any(c.symbol_name == "foo" for c in changes)

    def test_detect_parameter_added(self, detector):
        old = "def foo(x):\n    pass\n"
        new = "def foo(x, y):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.PARAMETER_ADDED for c in changes)

    def test_detect_parameter_removed(self, detector):
        old = "def foo(x, y):\n    pass\n"
        new = "def foo(x):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.PARAMETER_REMOVED for c in changes)

    def test_detect_return_type_changed(self, detector):
        old = "def foo(x) -> int:\n    return x\n"
        new = "def foo(x) -> str:\n    return str(x)\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.RETURN_TYPE_CHANGED for c in changes)

    def test_detect_class_removed(self, detector):
        old = "class Foo:\n    pass\n"
        new = "# Foo removed\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.CLASS_REMOVED for c in changes)
        assert any(c.symbol_name == "Foo" for c in changes)

    def test_detect_class_renamed(self, detector):
        old = "class Foo(Base):\n    pass\n"
        new = "class Bar(Base):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.CLASS_RENAMED for c in changes)

    def test_detect_constant_changed(self, detector):
        old = "MAX_SIZE = 100\n"
        new = "MAX_SIZE = 200\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.CONSTANT_CHANGED for c in changes)
        assert any("100" in c.old_signature and "200" in c.new_signature for c in changes)

    def test_detect_constant_removed(self, detector):
        old = "MAX_SIZE = 100\n"
        new = "# removed\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.CONSTANT_CHANGED for c in changes)

    def test_no_changes_same_content(self, detector):
        code = "def foo(x):\n    return x\n"
        changes = detector.detect_changes(code, code, "test.py")
        assert len(changes) == 0

    def test_detect_method_changed(self, detector):
        old = "class Foo:\n    def bar(self, x):\n        pass\n"
        new = "class Foo:\n    def bar(self, x, y):\n        pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.METHOD_CHANGED for c in changes)

    def test_detect_multiple_changes(self, detector):
        old = "def foo(x):\n    pass\n\ndef bar(y):\n    pass\n"
        new = "def foo(x, z):\n    pass\n\n# bar removed\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert len(changes) >= 2

    def test_detect_function_renamed(self, detector):
        old = "def calculate(x, y):\n    return x + y\n"
        new = "def compute(x, y):\n    return x + y\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(
            c.change_type == ChangeType.FUNCTION_RENAMED
            for c in changes
        )

    def test_parameter_renamed(self, detector):
        old = "def foo(old_name):\n    pass\n"
        new = "def foo(new_name):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.change_type == ChangeType.PARAMETER_RENAMED for c in changes)


# ---------------------------------------------------------------------------
# compare_signatures tests
# ---------------------------------------------------------------------------

class TestCompareSignatures:
    def test_identical_signatures(self, detector):
        diffs = detector.compare_signatures("def foo(x)", "def foo(x)")
        assert diffs == []

    def test_name_change(self, detector):
        diffs = detector.compare_signatures("def foo(x)", "def bar(x)")
        assert any("name" in d for d in diffs)

    def test_param_added(self, detector):
        diffs = detector.compare_signatures("def foo(x)", "def foo(x, y)")
        assert any("added" in d for d in diffs)

    def test_param_removed(self, detector):
        diffs = detector.compare_signatures("def foo(x, y)", "def foo(x)")
        assert any("removed" in d for d in diffs)

    def test_return_type_changed(self, detector):
        diffs = detector.compare_signatures("def foo(x) -> int", "def foo(x) -> str")
        assert any("return" in d.lower() for d in diffs)

    def test_annotation_changed(self, detector):
        diffs = detector.compare_signatures("def foo(x: int)", "def foo(x: str)")
        assert any("changed" in d for d in diffs)


# ---------------------------------------------------------------------------
# ImpactAnalyzer tests
# ---------------------------------------------------------------------------

class TestImpactAnalyzer:
    def test_find_impacted_files(self, analyzer, sample_project):
        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path=str(sample_project / "main.py"),
                symbol_name="process_data",
                old_signature="def process_data(data, format='json')",
                new_signature="",
                line_number=1,
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(sample_project))
        assert len(impacts) >= 1
        paths = [i.file_path for i in impacts]
        assert any("caller.py" in p for p in paths)

    def test_risk_high_for_callers(self, analyzer, sample_project):
        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path=str(sample_project / "main.py"),
                symbol_name="process_data",
                old_signature="def process_data(data, format='json')",
                new_signature="",
                line_number=1,
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(sample_project))
        caller_impact = [i for i in impacts if "caller.py" in i.file_path]
        assert len(caller_impact) >= 1
        assert caller_impact[0].risk_level == "high"

    def test_inheritance_detected(self, analyzer, sample_project):
        changes = [
            DetectedChange(
                change_type=ChangeType.CLASS_REMOVED,
                file_path=str(sample_project / "main.py"),
                symbol_name="DataProcessor",
                old_signature="class DataProcessor",
                new_signature="",
                line_number=4,
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(sample_project))
        usages = []
        for impact in impacts:
            for u in impact.usages:
                usages.append(u)
        assert any(u.usage_type == "inheritance" for u in usages)

    def test_type_hint_detected(self, analyzer, sample_project):
        changes = [
            DetectedChange(
                change_type=ChangeType.CLASS_RENAMED,
                file_path=str(sample_project / "main.py"),
                symbol_name="DataProcessor",
                old_signature="class DataProcessor",
                new_signature="class NewProcessor",
                line_number=4,
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(sample_project))
        usages = []
        for impact in impacts:
            for u in impact.usages:
                usages.append(u)
        assert any(u.usage_type == "type_hint" for u in usages)

    def test_import_detected(self, analyzer, sample_project):
        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path=str(sample_project / "main.py"),
                symbol_name="process_data",
                old_signature="def process_data(data)",
                new_signature="",
                line_number=1,
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(sample_project))
        usages = []
        for impact in impacts:
            for u in impact.usages:
                usages.append(u)
        assert any(u.usage_type == "import" for u in usages)

    def test_no_impact_empty_project(self, analyzer, tmp_path):
        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path="test.py",
                symbol_name="nonexistent",
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(tmp_path))
        assert impacts == []

    def test_no_impact_empty_changes(self, analyzer, sample_project):
        impacts = analyzer.analyze_impact([], str(sample_project))
        assert impacts == []

    def test_suggest_updates_function_renamed(self, analyzer):
        change = DetectedChange(
            change_type=ChangeType.FUNCTION_RENAMED,
            file_path="test.py",
            symbol_name="old_name",
            old_signature="def old_name(x)",
            new_signature="def new_name(x)",
        )
        usage = Usage(line_number=5, line_content="old_name(42)", usage_type="call")
        suggestion = analyzer.suggest_updates(change, usage)
        assert "new_name" in suggestion

    def test_suggest_updates_parameter_added(self, analyzer):
        change = DetectedChange(
            change_type=ChangeType.PARAMETER_ADDED,
            file_path="test.py",
            symbol_name="foo",
            old_signature="def foo(x)",
            new_signature="def foo(x, y)",
        )
        usage = Usage(line_number=10, line_content="foo(1)", usage_type="call")
        suggestion = analyzer.suggest_updates(change, usage)
        assert "new parameter" in suggestion.lower() or "added" in suggestion.lower()

    def test_suggest_updates_class_renamed(self, analyzer):
        change = DetectedChange(
            change_type=ChangeType.CLASS_RENAMED,
            file_path="test.py",
            symbol_name="OldClass",
            old_signature="class OldClass",
            new_signature="class NewClass",
        )
        usage = Usage(line_number=3, line_content="x = OldClass()", usage_type="call")
        suggestion = analyzer.suggest_updates(change, usage)
        assert "NewClass" in suggestion


# ---------------------------------------------------------------------------
# ImpactReport tests
# ---------------------------------------------------------------------------

class TestImpactReport:
    def _make_report_data(self):
        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path="main.py",
                symbol_name="foo",
                old_signature="def foo(x)",
                new_signature="",
                line_number=1,
            ),
            DetectedChange(
                change_type=ChangeType.PARAMETER_ADDED,
                file_path="main.py",
                symbol_name="bar",
                old_signature="def bar(x)",
                new_signature="def bar(x, y)",
                line_number=5,
            ),
        ]
        impacts = [
            ImpactedFile(
                file_path="caller.py",
                usages=[
                    Usage(line_number=3, line_content="foo(1)", usage_type="call"),
                    Usage(line_number=1, line_content="from main import foo", usage_type="import"),
                ],
                risk_level="high",
                suggested_update="Remove usage of foo",
            ),
            ImpactedFile(
                file_path="other.py",
                usages=[
                    Usage(line_number=10, line_content="bar(1)", usage_type="call"),
                ],
                risk_level="medium",
                suggested_update="Update call to bar",
            ),
        ]
        return changes, impacts

    def test_generate_report(self):
        changes, impacts = self._make_report_data()
        report = ImpactReport(changes, impacts)
        text = report.generate()
        assert "CHANGE IMPACT ANALYSIS REPORT" in text
        assert "foo" in text
        assert "bar" in text
        assert "caller.py" in text
        assert "HIGH RISK" in text

    def test_generate_prompt(self):
        changes, impacts = self._make_report_data()
        report = ImpactReport(changes, impacts)
        prompt = report.generate_prompt()
        assert "update" in prompt.lower()
        assert "foo" in prompt
        assert "bar" in prompt
        assert "caller.py" in prompt

    def test_summary(self):
        changes, impacts = self._make_report_data()
        report = ImpactReport(changes, impacts)
        s = report.summary()
        assert s["total_changes"] == 2
        assert s["total_impacted"] == 2
        assert s["high"] == 1
        assert s["medium"] == 1
        assert s["low"] == 0

    def test_empty_report(self):
        report = ImpactReport()
        text = report.generate()
        assert "Changes detected: 0" in text
        s = report.summary()
        assert s["total_changes"] == 0

    def test_generate_with_args(self):
        """Test generate() with explicit args overriding constructor."""
        report = ImpactReport()
        changes, impacts = self._make_report_data()
        text = report.generate(changes, impacts)
        assert "foo" in text

    def test_prompt_contains_file_paths(self):
        changes, impacts = self._make_report_data()
        report = ImpactReport(changes, impacts)
        prompt = report.generate_prompt()
        assert "caller.py" in prompt
        assert "other.py" in prompt

    def test_report_sorts_by_risk(self):
        changes, impacts = self._make_report_data()
        report = ImpactReport(changes, impacts)
        text = report.generate()
        # HIGH RISK should appear before MEDIUM RISK
        high_pos = text.index("HIGH RISK")
        medium_pos = text.index("MEDIUM RISK")
        assert high_pos < medium_pos


# ---------------------------------------------------------------------------
# quick_impact_check tests
# ---------------------------------------------------------------------------

class TestQuickImpactCheck:
    def test_quick_check_returns_report(self, sample_project):
        old = "def process_data(data, format='json'):\n    return data\n"
        new = "def process_data(data, format='json', verbose=False):\n    return data\n"
        report = quick_impact_check(
            old, new,
            str(sample_project / "main.py"),
            str(sample_project),
        )
        assert isinstance(report, ImpactReport)
        assert len(report.changes) >= 1

    def test_quick_check_with_removal(self, sample_project):
        old = "def process_data(data, format='json'):\n    return data\n"
        new = "# removed\n"
        report = quick_impact_check(
            old, new,
            str(sample_project / "main.py"),
            str(sample_project),
        )
        assert len(report.changes) >= 1
        assert len(report.impacts) >= 1

    def test_quick_check_no_changes(self, sample_project):
        code = "def process_data(data, format='json'):\n    return data\n"
        report = quick_impact_check(
            code, code,
            str(sample_project / "main.py"),
            str(sample_project),
        )
        assert len(report.changes) == 0
        assert len(report.impacts) == 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_syntax_error_in_old(self, detector):
        old = "def foo(:\n"
        new = "def bar(x):\n    pass\n"
        # Should not crash — falls back to regex
        changes = detector.detect_changes(old, new, "test.py")
        # May or may not detect changes, but should not raise
        assert isinstance(changes, list)

    def test_empty_content(self, detector):
        changes = detector.detect_changes("", "", "test.py")
        assert changes == []

    def test_async_function(self, detector):
        old = "async def fetch(url):\n    pass\n"
        new = "async def fetch(url, timeout=30):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert len(changes) >= 1

    def test_multiple_classes(self, detector):
        old = "class A:\n    pass\n\nclass B:\n    pass\n"
        new = "class A:\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert any(c.symbol_name == "B" for c in changes)

    def test_skip_pycache(self, analyzer, tmp_path):
        """__pycache__ directories should be skipped."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("import foo\nfoo()\n")

        changes = [
            DetectedChange(
                change_type=ChangeType.FUNCTION_REMOVED,
                file_path="other.py",
                symbol_name="foo",
            )
        ]
        impacts = analyzer.analyze_impact(changes, str(tmp_path))
        paths = [i.file_path for i in impacts]
        assert not any("__pycache__" in p for p in paths)

    def test_unicode_content(self, detector):
        old = "def grüße(name):\n    return f'Hallo {name}'\n"
        new = "def grüße(name, formal=False):\n    return f'Hallo {name}'\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert len(changes) >= 1

    def test_decorator_functions(self, detector):
        old = "@decorator\ndef foo(x):\n    pass\n"
        new = "@decorator\ndef foo(x, y):\n    pass\n"
        changes = detector.detect_changes(old, new, "test.py")
        assert len(changes) >= 1
        assert any(c.symbol_name == "foo" for c in changes)
