"""
Tests for edit_format module — Edit Format Optimization.

30+ tests covering EditFormat, EditFormatSelector, FormatGenerator,
FormatParser, FormatTracker, EditOperation, and EditRegion.
"""

import os
import sys
import unittest

# Ensure new-files directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from edit_format import (
    EditFormat,
    EditFormatSelector,
    EditOperation,
    EditRegion,
    FormatGenerator,
    FormatParser,
    FormatTracker,
)

# ─── EditFormat enum tests ──────────────────────────────────────────────────


class TestEditFormat(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(EditFormat.SEARCH_REPLACE.value, "search_replace")
        self.assertEqual(EditFormat.WHOLE_FILE.value, "whole_file")
        self.assertEqual(EditFormat.UNIFIED_DIFF.value, "unified_diff")
        self.assertEqual(EditFormat.XML_TAGGED.value, "xml_tagged")

    def test_enum_members_count(self):
        self.assertEqual(len(EditFormat), 4)

    def test_enum_from_value(self):
        self.assertEqual(EditFormat("search_replace"), EditFormat.SEARCH_REPLACE)
        self.assertEqual(EditFormat("xml_tagged"), EditFormat.XML_TAGGED)


# ─── Dataclass tests ────────────────────────────────────────────────────────


class TestEditOperation(unittest.TestCase):
    def test_basic_creation(self):
        op = EditOperation(file_path="test.py", old_content="old", new_content="new")
        self.assertEqual(op.file_path, "test.py")
        self.assertEqual(op.old_content, "old")
        self.assertEqual(op.new_content, "new")
        self.assertIsNone(op.start_line)
        self.assertIsNone(op.end_line)

    def test_with_line_numbers(self):
        op = EditOperation("f.py", "a", "b", start_line=10, end_line=20)
        self.assertEqual(op.start_line, 10)
        self.assertEqual(op.end_line, 20)


class TestEditRegion(unittest.TestCase):
    def test_creation(self):
        region = EditRegion(
            label="fix import",
            start_line=1,
            end_line=5,
            old_content="import os",
            new_content="import sys",
        )
        self.assertEqual(region.label, "fix import")
        self.assertEqual(region.start_line, 1)
        self.assertEqual(region.end_line, 5)


# ─── EditFormatSelector tests ───────────────────────────────────────────────


class TestEditFormatSelector(unittest.TestCase):
    def setUp(self):
        self.selector = EditFormatSelector()

    def test_small_file_returns_whole_file(self):
        fmt = self.selector.select_format("app.py", "add logging", 45, "single_function")
        self.assertEqual(fmt, EditFormat.WHOLE_FILE)

    def test_small_file_boundary(self):
        fmt = self.selector.select_format("app.py", "edit", 79, "single_line")
        self.assertEqual(fmt, EditFormat.WHOLE_FILE)

    def test_at_threshold_returns_search_replace(self):
        fmt = self.selector.select_format("app.py", "edit", 80, "single_line")
        self.assertEqual(fmt, EditFormat.SEARCH_REPLACE)

    def test_whole_file_scope(self):
        fmt = self.selector.select_format("app.py", "rewrite", 200, "whole_file")
        self.assertEqual(fmt, EditFormat.WHOLE_FILE)

    def test_multi_region_scope(self):
        fmt = self.selector.select_format("app.py", "edit", 200, "multi_region")
        self.assertEqual(fmt, EditFormat.XML_TAGGED)

    def test_complex_description_returns_xml(self):
        fmt = self.selector.select_format(
            "app.py", "complex restructure of the module", 200, "multi_function"
        )
        self.assertEqual(fmt, EditFormat.XML_TAGGED)

    def test_large_file_multi_function_returns_diff(self):
        fmt = self.selector.select_format(
            "app.py", "update handlers", 500, "multi_function"
        )
        self.assertEqual(fmt, EditFormat.UNIFIED_DIFF)

    def test_large_file_with_scattered_keyword(self):
        fmt = self.selector.select_format(
            "app.py", "fix multiple scattered bugs", 400, "single_function"
        )
        self.assertEqual(fmt, EditFormat.UNIFIED_DIFF)

    def test_medium_file_single_function_returns_search_replace(self):
        fmt = self.selector.select_format(
            "app.py", "fix bug in handler", 150, "single_function"
        )
        self.assertEqual(fmt, EditFormat.SEARCH_REPLACE)

    def test_custom_threshold(self):
        selector = EditFormatSelector(whole_file_threshold=50)
        fmt = selector.select_format("app.py", "edit", 60, "single_line")
        self.assertEqual(fmt, EditFormat.SEARCH_REPLACE)

    def test_custom_large_threshold(self):
        selector = EditFormatSelector(large_file_threshold=100)
        fmt = selector.select_format("app.py", "update", 150, "multi_function")
        self.assertEqual(fmt, EditFormat.UNIFIED_DIFF)

    def test_refactor_keyword_returns_xml(self):
        fmt = self.selector.select_format(
            "app.py", "refactor the authentication module", 200, "multi_function"
        )
        self.assertEqual(fmt, EditFormat.XML_TAGGED)


# ─── FormatGenerator tests ──────────────────────────────────────────────────


class TestFormatGenerator(unittest.TestCase):
    def test_generate_search_replace(self):
        result = FormatGenerator.generate_search_replace("old code", "new code")
        self.assertIn("<<<<<<< SEARCH", result)
        self.assertIn("old code", result)
        self.assertIn("=======", result)
        self.assertIn("new code", result)
        self.assertIn(">>>>>>> REPLACE", result)

    def test_generate_search_replace_multiline(self):
        old = "line1\nline2\nline3"
        new = "line1\nmodified\nline3"
        result = FormatGenerator.generate_search_replace(old, new)
        self.assertIn("line1\nline2\nline3", result)
        self.assertIn("line1\nmodified\nline3", result)

    def test_generate_whole_file(self):
        content = "import os\n\ndef main():\n    pass\n"
        result = FormatGenerator.generate_whole_file(content, "main.py")
        self.assertIn("--- main.py", result)
        self.assertIn("```", result)
        self.assertIn("import os", result)

    def test_generate_unified_diff(self):
        original = "line1\nline2\nline3\n"
        modified = "line1\nmodified\nline3\n"
        result = FormatGenerator.generate_unified_diff(original, modified, "test.py")
        self.assertIn("a/test.py", result)
        self.assertIn("b/test.py", result)
        self.assertIn("@@", result)

    def test_generate_unified_diff_no_changes(self):
        text = "same\n"
        result = FormatGenerator.generate_unified_diff(text, text, "test.py")
        self.assertEqual(result, "")

    def test_generate_xml_tagged(self):
        edits = [
            EditRegion("fix import", 1, 3, "import os", "import sys"),
            EditRegion("add function", 10, 15, "pass", "def hello():\n    print('hi')"),
        ]
        result = FormatGenerator.generate_xml_tagged(edits)
        self.assertIn("<edits>", result)
        self.assertIn("</edits>", result)
        self.assertIn("<label>fix import</label>", result)
        self.assertIn('<lines start="1" end="3" />', result)
        self.assertIn("import sys", result)
        self.assertIn("<label>add function</label>", result)

    def test_generate_xml_tagged_empty(self):
        result = FormatGenerator.generate_xml_tagged([])
        self.assertEqual(result, "<edits>\n</edits>")


# ─── FormatParser tests ─────────────────────────────────────────────────────


class TestFormatParser(unittest.TestCase):
    def test_parse_search_replace_single(self):
        text = (
            "<<<<<<< SEARCH\n"
            "old code here\n"
            "=======\n"
            "new code here\n"
            ">>>>>>> REPLACE"
        )
        ops = FormatParser.parse_search_replace(text)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].old_content, "old code here")
        self.assertEqual(ops[0].new_content, "new code here")

    def test_parse_search_replace_multiple(self):
        text = (
            "<<<<<<< SEARCH\nold1\n=======\nnew1\n>>>>>>> REPLACE\n\n"
            "<<<<<<< SEARCH\nold2\n=======\nnew2\n>>>>>>> REPLACE"
        )
        ops = FormatParser.parse_search_replace(text)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0].old_content, "old1")
        self.assertEqual(ops[1].old_content, "old2")

    def test_parse_search_replace_multiline_content(self):
        text = (
            "<<<<<<< SEARCH\n"
            "line1\nline2\nline3\n"
            "=======\n"
            "new1\nnew2\n"
            ">>>>>>> REPLACE"
        )
        ops = FormatParser.parse_search_replace(text)
        self.assertEqual(len(ops), 1)
        self.assertIn("line2", ops[0].old_content)

    def test_parse_search_replace_no_match(self):
        ops = FormatParser.parse_search_replace("just some text")
        self.assertEqual(len(ops), 0)

    def test_parse_unified_diff(self):
        diff_text = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-old line\n"
            "+new line\n"
            " line3\n"
        )
        ops = FormatParser.parse_unified_diff(diff_text)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].file_path, "test.py")
        self.assertIn("old line", ops[0].old_content)
        self.assertIn("new line", ops[0].new_content)
        self.assertEqual(ops[0].start_line, 1)

    def test_parse_unified_diff_multiple_hunks(self):
        diff_text = (
            "--- a/test.py\n"
            "+++ b/test.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-old1\n"
            "+new1\n"
            " line3\n"
            "@@ -10,3 +10,3 @@\n"
            " line10\n"
            "-old2\n"
            "+new2\n"
            " line12\n"
        )
        ops = FormatParser.parse_unified_diff(diff_text)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0].start_line, 1)
        self.assertEqual(ops[1].start_line, 10)

    def test_parse_xml_tagged(self):
        xml_text = (
            "<edits>\n"
            "  <edit>\n"
            "    <label>fix import</label>\n"
            '    <lines start="1" end="3" />\n'
            "    <old>\n"
            "      import os\n"
            "    </old>\n"
            "    <new>\n"
            "      import sys\n"
            "    </new>\n"
            "  </edit>\n"
            "</edits>"
        )
        ops = FormatParser.parse_xml_tagged(xml_text)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].old_content, "import os")
        self.assertEqual(ops[0].new_content, "import sys")
        self.assertEqual(ops[0].start_line, 1)
        self.assertEqual(ops[0].end_line, 3)

    def test_parse_xml_tagged_multiple(self):
        xml_text = (
            "<edits>\n"
            "  <edit>\n"
            "    <label>edit1</label>\n"
            '    <lines start="1" end="2" />\n'
            "    <old>\n"
            "      old1\n"
            "    </old>\n"
            "    <new>\n"
            "      new1\n"
            "    </new>\n"
            "  </edit>\n"
            "  <edit>\n"
            "    <label>edit2</label>\n"
            '    <lines start="10" end="12" />\n'
            "    <old>\n"
            "      old2\n"
            "    </old>\n"
            "    <new>\n"
            "      new2\n"
            "    </new>\n"
            "  </edit>\n"
            "</edits>"
        )
        ops = FormatParser.parse_xml_tagged(xml_text)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[1].start_line, 10)

    def test_parse_whole_file(self):
        text = "--- main.py\n```\nimport os\n\ndef main():\n    pass\n```"
        ops = FormatParser.parse_whole_file(text)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].file_path, "main.py")
        self.assertIn("import os", ops[0].new_content)

    def test_parse_whole_file_no_match(self):
        ops = FormatParser.parse_whole_file("just text")
        self.assertEqual(len(ops), 0)


# ─── Auto-detect format tests ───────────────────────────────────────────────


class TestAutoDetectFormat(unittest.TestCase):
    def test_detect_search_replace(self):
        text = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.SEARCH_REPLACE)

    def test_detect_xml_tagged(self):
        text = "<edits>\n<edit>\n<label>x</label>\n</edit>\n</edits>"
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.XML_TAGGED)

    def test_detect_unified_diff(self):
        text = "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,3 @@\n line1\n-old\n+new\n"
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.UNIFIED_DIFF)

    def test_detect_whole_file(self):
        text = "--- main.py\n```\ncontent\n```"
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.WHOLE_FILE)

    def test_detect_fallback(self):
        text = "just some plain text with no edit markers"
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.SEARCH_REPLACE)


# ─── FormatTracker tests ────────────────────────────────────────────────────


class TestFormatTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = FormatTracker(min_samples=3)

    def test_empty_tracker_recommends_default(self):
        fmt = self.tracker.recommend_format("gpt-4")
        self.assertEqual(fmt, EditFormat.SEARCH_REPLACE)

    def test_record_success(self):
        self.tracker.record_success("gpt-4", EditFormat.SEARCH_REPLACE)
        self.assertEqual(self.tracker.get_total_attempts("gpt-4", EditFormat.SEARCH_REPLACE), 1)
        self.assertEqual(self.tracker.get_success_rate("gpt-4", EditFormat.SEARCH_REPLACE), 1.0)

    def test_record_failure(self):
        self.tracker.record_failure("gpt-4", EditFormat.UNIFIED_DIFF)
        self.assertEqual(self.tracker.get_total_attempts("gpt-4", EditFormat.UNIFIED_DIFF), 1)
        self.assertEqual(self.tracker.get_success_rate("gpt-4", EditFormat.UNIFIED_DIFF), 0.0)

    def test_mixed_success_rate(self):
        for _ in range(3):
            self.tracker.record_success("claude-3", EditFormat.WHOLE_FILE)
        self.tracker.record_failure("claude-3", EditFormat.WHOLE_FILE)
        rate = self.tracker.get_success_rate("claude-3", EditFormat.WHOLE_FILE)
        self.assertAlmostEqual(rate, 0.75)

    def test_recommend_best_format(self):
        # WHOLE_FILE: 4/5 = 80%
        for _ in range(4):
            self.tracker.record_success("gpt-4", EditFormat.WHOLE_FILE)
        self.tracker.record_failure("gpt-4", EditFormat.WHOLE_FILE)

        # SEARCH_REPLACE: 2/3 = 67%
        for _ in range(2):
            self.tracker.record_success("gpt-4", EditFormat.SEARCH_REPLACE)
        self.tracker.record_failure("gpt-4", EditFormat.SEARCH_REPLACE)

        fmt = self.tracker.recommend_format("gpt-4")
        self.assertEqual(fmt, EditFormat.WHOLE_FILE)

    def test_recommend_ignores_insufficient_samples(self):
        # Only 2 samples, below min_samples=3
        self.tracker.record_success("gpt-4", EditFormat.XML_TAGGED)
        self.tracker.record_success("gpt-4", EditFormat.XML_TAGGED)

        # 3 samples for SEARCH_REPLACE
        for _ in range(3):
            self.tracker.record_success("gpt-4", EditFormat.SEARCH_REPLACE)

        fmt = self.tracker.recommend_format("gpt-4")
        self.assertEqual(fmt, EditFormat.SEARCH_REPLACE)

    def test_get_success_rate_unknown_model(self):
        rate = self.tracker.get_success_rate("unknown", EditFormat.SEARCH_REPLACE)
        self.assertEqual(rate, 0.0)

    def test_get_total_attempts_unknown(self):
        total = self.tracker.get_total_attempts("unknown", EditFormat.SEARCH_REPLACE)
        self.assertEqual(total, 0)

    def test_get_summary(self):
        self.tracker.record_success("gpt-4", EditFormat.SEARCH_REPLACE)
        self.tracker.record_failure("gpt-4", EditFormat.SEARCH_REPLACE)
        summary = self.tracker.get_summary("gpt-4")
        self.assertIn("search_replace", summary)
        self.assertAlmostEqual(summary["search_replace"]["success_rate"], 0.5)
        self.assertEqual(summary["search_replace"]["total"], 2.0)

    def test_get_summary_empty_model(self):
        summary = self.tracker.get_summary("nonexistent")
        self.assertEqual(summary, {})

    def test_multiple_models_independent(self):
        self.tracker.record_success("gpt-4", EditFormat.SEARCH_REPLACE)
        self.tracker.record_failure("claude-3", EditFormat.SEARCH_REPLACE)
        self.assertEqual(self.tracker.get_success_rate("gpt-4", EditFormat.SEARCH_REPLACE), 1.0)
        self.assertEqual(self.tracker.get_success_rate("claude-3", EditFormat.SEARCH_REPLACE), 0.0)


# ─── Round-trip tests (generate -> parse) ────────────────────────────────────


class TestRoundTrip(unittest.TestCase):
    def test_search_replace_roundtrip(self):
        old = "def hello():\n    print('hello')"
        new = "def hello():\n    print('goodbye')"
        generated = FormatGenerator.generate_search_replace(old, new)
        ops = FormatParser.parse_search_replace(generated)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].old_content, old)
        self.assertEqual(ops[0].new_content, new)

    def test_xml_roundtrip(self):
        edits = [
            EditRegion("test", 5, 10, "old_code", "new_code"),
        ]
        generated = FormatGenerator.generate_xml_tagged(edits)
        ops = FormatParser.parse_xml_tagged(generated)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].old_content, "old_code")
        self.assertEqual(ops[0].new_content, "new_code")
        self.assertEqual(ops[0].start_line, 5)
        self.assertEqual(ops[0].end_line, 10)

    def test_whole_file_roundtrip(self):
        content = "import os\n\ndef main():\n    pass"
        generated = FormatGenerator.generate_whole_file(content, "test.py")
        ops = FormatParser.parse_whole_file(generated)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].file_path, "test.py")
        self.assertEqual(ops[0].new_content, content)

    def test_auto_detect_on_generated_search_replace(self):
        text = FormatGenerator.generate_search_replace("a", "b")
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.SEARCH_REPLACE)

    def test_auto_detect_on_generated_xml(self):
        text = FormatGenerator.generate_xml_tagged([
            EditRegion("x", 1, 2, "a", "b"),
        ])
        self.assertEqual(FormatParser.auto_detect_format(text), EditFormat.XML_TAGGED)


if __name__ == "__main__":
    unittest.main()
