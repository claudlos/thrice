"""Tests for the Reproduce-First Debugging workflow module."""

import sys
import os
import unittest

# Add the new-files directory to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'new-files'))

from reproduce_first import (
    BugAnalyzer,
    BugType,
    FileLocation,
    ReproductionResult,
    ReproduceFirstWorkflow,
    detect_bug_report,
    generate_debug_prompt,
    _generate_reproduction_script,
    _suggest_fix_approach,
)


# ─── Sample Error Texts ───────────────────────────────────────────────────

SAMPLE_TRACEBACK = """\
Traceback (most recent call last):
  File "/home/user/project/app.py", line 42, in main
    result = process_data(data)
  File "/home/user/project/utils.py", line 15, in process_data
    return data["key"] + 1
TypeError: can only concatenate str (not "int") to str
"""

SAMPLE_IMPORT_ERROR = """\
Traceback (most recent call last):
  File "/home/user/project/app.py", line 1, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'
"""

SAMPLE_SYNTAX_ERROR = """\
  File "/home/user/project/broken.py", line 10
    def foo(
           ^
SyntaxError: unexpected EOF while parsing
"""

SAMPLE_ASSERTION_ERROR = """\
FAILED tests/test_app.py::test_addition
AssertionError: assert 3 == 4
"""

SAMPLE_TIMEOUT_ERROR = """\
TimeoutError: Operation timed out after 30 seconds
"""

SAMPLE_PERMISSION_ERROR = """\
PermissionError: [Errno 13] Permission denied: '/etc/shadow'
"""

SAMPLE_LOGIC_BUG = """\
The function returns wrong output. Expected 42 but got 0 instead.
"""


# ─── Test detect_bug_report ───────────────────────────────────────────────


class TestDetectBugReport(unittest.TestCase):
    """Tests for the detect_bug_report() function."""

    def test_detects_traceback(self):
        self.assertTrue(detect_bug_report(SAMPLE_TRACEBACK))

    def test_detects_error_keyword(self):
        self.assertTrue(detect_bug_report("I'm getting an error when I run the script"))

    def test_detects_bug_keyword(self):
        self.assertTrue(detect_bug_report("There's a bug in the login function"))

    def test_detects_broken_keyword(self):
        self.assertTrue(detect_bug_report("The API endpoint is broken"))

    def test_detects_crash_keyword(self):
        self.assertTrue(detect_bug_report("The app crashes on startup"))

    def test_detects_fails_keyword(self):
        self.assertTrue(detect_bug_report("The test fails intermittently"))

    def test_detects_doesnt_work(self):
        self.assertTrue(detect_bug_report("This function doesn't work"))

    def test_detects_stack_trace_mention(self):
        self.assertTrue(detect_bug_report("Here's the stack trace from the crash"))

    def test_detects_exit_code(self):
        self.assertTrue(detect_bug_report("Process exited with exit code 1"))

    def test_no_bug_in_normal_message(self):
        self.assertFalse(detect_bug_report("Please add a new feature to the app"))

    def test_no_bug_in_question(self):
        self.assertFalse(detect_bug_report("How do I create a new class?"))

    def test_empty_message(self):
        self.assertFalse(detect_bug_report(""))

    def test_none_like_empty(self):
        self.assertFalse(detect_bug_report("   "))

    def test_detects_import_error_text(self):
        self.assertTrue(detect_bug_report(SAMPLE_IMPORT_ERROR))

    def test_detects_syntax_error_text(self):
        self.assertTrue(detect_bug_report(SAMPLE_SYNTAX_ERROR))


# ─── Test BugType Enum ────────────────────────────────────────────────────


class TestBugType(unittest.TestCase):
    """Tests for the BugType enum."""

    def test_all_values_are_strings(self):
        for bt in BugType:
            self.assertIsInstance(bt.value, str)

    def test_expected_members_exist(self):
        expected = {"SYNTAX", "IMPORT", "TYPE", "LOGIC", "RUNTIME", "TIMEOUT",
                    "ASSERTION", "PERMISSION", "NOT_FOUND", "UNKNOWN"}
        actual = {bt.name for bt in BugType}
        self.assertEqual(expected, actual)


# ─── Test FileLocation ────────────────────────────────────────────────────


class TestFileLocation(unittest.TestCase):
    """Tests for the FileLocation dataclass."""

    def test_str_without_function(self):
        loc = FileLocation(filepath="/app.py", line_number=10)
        self.assertEqual(str(loc), "/app.py:10")

    def test_str_with_function(self):
        loc = FileLocation(filepath="/app.py", line_number=10, function_name="main")
        self.assertEqual(str(loc), "/app.py:10 in main")


# ─── Test ReproductionResult ──────────────────────────────────────────────


class TestReproductionResult(unittest.TestCase):
    """Tests for the ReproductionResult dataclass."""

    def test_summary_confirmed(self):
        result = ReproductionResult(
            bug_confirmed=True,
            reproduction_script="print('test')",
            error_output="TypeError: bad",
            suggested_fix_approach="Fix the types",
            bug_type=BugType.TYPE,
        )
        summary = result.summary()
        self.assertIn("CONFIRMED", summary)
        self.assertIn("type", summary)

    def test_summary_not_confirmed(self):
        result = ReproductionResult(
            bug_confirmed=False,
            reproduction_script="",
            error_output="",
            suggested_fix_approach="",
        )
        summary = result.summary()
        self.assertIn("NOT CONFIRMED", summary)

    def test_summary_truncates_long_error(self):
        result = ReproductionResult(
            bug_confirmed=True,
            reproduction_script="",
            error_output="x" * 300,
            suggested_fix_approach="",
        )
        summary = result.summary()
        self.assertIn("...", summary)

    def test_summary_includes_locations(self):
        loc = FileLocation(filepath="/app.py", line_number=5, function_name="foo")
        result = ReproductionResult(
            bug_confirmed=True,
            reproduction_script="",
            error_output="error",
            suggested_fix_approach="fix it",
            file_locations=[loc],
        )
        summary = result.summary()
        self.assertIn("/app.py:5 in foo", summary)


# ─── Test BugAnalyzer ─────────────────────────────────────────────────────


class TestBugAnalyzer(unittest.TestCase):
    """Tests for the BugAnalyzer class."""

    def setUp(self):
        self.analyzer = BugAnalyzer()

    def test_classify_type_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_TRACEBACK),
            BugType.TYPE,
        )

    def test_classify_import_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_IMPORT_ERROR),
            BugType.IMPORT,
        )

    def test_classify_syntax_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_SYNTAX_ERROR),
            BugType.SYNTAX,
        )

    def test_classify_assertion_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_ASSERTION_ERROR),
            BugType.ASSERTION,
        )

    def test_classify_timeout_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_TIMEOUT_ERROR),
            BugType.TIMEOUT,
        )

    def test_classify_permission_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_PERMISSION_ERROR),
            BugType.PERMISSION,
        )

    def test_classify_logic_error(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(SAMPLE_LOGIC_BUG),
            BugType.LOGIC,
        )

    def test_classify_unknown(self):
        self.assertEqual(
            self.analyzer.classify_bug_type("something went haywire"),
            BugType.UNKNOWN,
        )

    def test_classify_empty(self):
        self.assertEqual(
            self.analyzer.classify_bug_type(""),
            BugType.UNKNOWN,
        )

    def test_extract_file_locations(self):
        locations = self.analyzer.extract_file_locations(SAMPLE_TRACEBACK)
        self.assertEqual(len(locations), 2)
        self.assertEqual(locations[0].filepath, "/home/user/project/app.py")
        self.assertEqual(locations[0].line_number, 42)
        self.assertEqual(locations[0].function_name, "main")
        self.assertEqual(locations[1].filepath, "/home/user/project/utils.py")
        self.assertEqual(locations[1].line_number, 15)

    def test_extract_file_locations_empty(self):
        self.assertEqual(self.analyzer.extract_file_locations("no traceback here"), [])

    def test_extract_error_message_from_traceback(self):
        msg = self.analyzer.extract_error_message(SAMPLE_TRACEBACK)
        self.assertIn("TypeError", msg)
        self.assertIn("can only concatenate", msg)

    def test_extract_error_message_empty(self):
        self.assertEqual(self.analyzer.extract_error_message(""), "")

    def test_extract_error_message_plain_text(self):
        msg = self.analyzer.extract_error_message("something broke")
        self.assertEqual(msg, "something broke")

    def test_suggest_files_filters_stdlib(self):
        error = """\
Traceback (most recent call last):
  File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/home/user/project/app.py", line 5, in main
    do_stuff()
ImportError: cannot import name 'missing'
"""
        files = self.analyzer.suggest_files_to_examine(error)
        self.assertTrue(len(files) >= 1)
        # User file should come first
        self.assertEqual(files[0], "/home/user/project/app.py")

    def test_suggest_files_empty(self):
        self.assertEqual(self.analyzer.suggest_files_to_examine("no files"), [])

    def test_analyze_returns_all_keys(self):
        result = self.analyzer.analyze(SAMPLE_TRACEBACK)
        self.assertIn("bug_type", result)
        self.assertIn("error_message", result)
        self.assertIn("file_locations", result)
        self.assertIn("files_to_examine", result)

    def test_suggest_files_deduplicates(self):
        error = """\
Traceback (most recent call last):
  File "/home/user/app.py", line 1, in main
    foo()
  File "/home/user/app.py", line 5, in foo
    bar()
RuntimeError: oops
"""
        files = self.analyzer.suggest_files_to_examine(error)
        self.assertEqual(files.count("/home/user/app.py"), 1)


# ─── Test ReproduceFirstWorkflow ──────────────────────────────────────────


class TestReproduceFirstWorkflow(unittest.TestCase):
    """Tests for the ReproduceFirstWorkflow class."""

    def setUp(self):
        self.workflow = ReproduceFirstWorkflow()

    def test_detect_bug_true(self):
        self.assertTrue(self.workflow.detect_bug(SAMPLE_TRACEBACK))

    def test_detect_bug_false(self):
        self.assertFalse(self.workflow.detect_bug("Add a feature please"))

    def test_generate_reproduction_returns_result(self):
        result = self.workflow.generate_reproduction(SAMPLE_TRACEBACK)
        self.assertIsInstance(result, ReproductionResult)
        self.assertFalse(result.bug_confirmed)  # Not yet confirmed
        self.assertTrue(len(result.reproduction_script) > 0)
        self.assertEqual(result.bug_type, BugType.TYPE)

    def test_generate_reproduction_import(self):
        result = self.workflow.generate_reproduction(SAMPLE_IMPORT_ERROR)
        self.assertEqual(result.bug_type, BugType.IMPORT)
        self.assertIn("nonexistent_module", result.reproduction_script)

    def test_analyze_error(self):
        analysis = self.workflow.analyze_error(SAMPLE_TRACEBACK)
        self.assertEqual(analysis["bug_type"], BugType.TYPE)
        self.assertTrue(len(analysis["file_locations"]) > 0)

    def test_suggest_approach(self):
        analysis = {"bug_type": BugType.SYNTAX, "error_message": "bad syntax"}
        approach = self.workflow.suggest_approach(analysis)
        self.assertIn("syntax", approach.lower())

    def test_suggest_approach_unknown(self):
        analysis = {"bug_type": BugType.UNKNOWN}
        approach = self.workflow.suggest_approach(analysis)
        self.assertIn("Investigate", approach)

    def test_run_returns_result_on_bug(self):
        result = self.workflow.run(SAMPLE_TRACEBACK)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ReproductionResult)

    def test_run_returns_none_on_no_bug(self):
        result = self.workflow.run("Please add a new button to the UI")
        self.assertIsNone(result)

    def test_run_with_import_error(self):
        result = self.workflow.run(SAMPLE_IMPORT_ERROR)
        self.assertIsNotNone(result)
        self.assertEqual(result.bug_type, BugType.IMPORT)


# ─── Test generate_debug_prompt ───────────────────────────────────────────


class TestGenerateDebugPrompt(unittest.TestCase):
    """Tests for the generate_debug_prompt() function."""

    def test_basic_prompt(self):
        prompt = generate_debug_prompt(SAMPLE_TRACEBACK)
        self.assertIn("Reproduce-First", prompt)
        self.assertIn("TypeError", prompt)
        self.assertIn("REPRODUCE", prompt)
        self.assertIn("VERIFY", prompt)

    def test_prompt_with_context(self):
        prompt = generate_debug_prompt(
            SAMPLE_TRACEBACK,
            context="This happens only with Unicode input",
        )
        self.assertIn("Unicode input", prompt)

    def test_prompt_with_previous_attempts(self):
        prompt = generate_debug_prompt(
            SAMPLE_TRACEBACK,
            previous_attempts=["Tried converting to int", "Tried str()"],
        )
        self.assertIn("Previous Fix Attempts", prompt)
        self.assertIn("Tried converting to int", prompt)
        self.assertIn("fundamentally different", prompt)

    def test_prompt_includes_files(self):
        prompt = generate_debug_prompt(SAMPLE_TRACEBACK)
        self.assertIn("/home/user/project/app.py", prompt)
        self.assertIn("/home/user/project/utils.py", prompt)

    def test_prompt_includes_bug_type(self):
        prompt = generate_debug_prompt(SAMPLE_IMPORT_ERROR)
        self.assertIn("import", prompt)


# ─── Test _suggest_fix_approach ───────────────────────────────────────────


class TestSuggestFixApproach(unittest.TestCase):
    """Tests for the _suggest_fix_approach() helper."""

    def test_each_bug_type_has_approach(self):
        for bug_type in BugType:
            approach = _suggest_fix_approach(bug_type, "some error")
            self.assertTrue(len(approach) > 0, f"No approach for {bug_type}")


# ─── Test _generate_reproduction_script ───────────────────────────────────


class TestGenerateReproductionScript(unittest.TestCase):
    """Tests for the _generate_reproduction_script() helper."""

    def test_import_script_includes_module(self):
        script = _generate_reproduction_script(SAMPLE_IMPORT_ERROR, BugType.IMPORT)
        self.assertIn("import nonexistent_module", script)

    def test_syntax_script_includes_compile(self):
        script = _generate_reproduction_script(SAMPLE_SYNTAX_ERROR, BugType.SYNTAX)
        self.assertIn("py_compile", script)

    def test_timeout_script_includes_signal(self):
        script = _generate_reproduction_script(SAMPLE_TIMEOUT_ERROR, BugType.TIMEOUT)
        self.assertIn("signal", script)

    def test_assertion_script_includes_pytest(self):
        script = _generate_reproduction_script(SAMPLE_ASSERTION_ERROR, BugType.ASSERTION)
        self.assertIn("pytest", script)

    def test_unknown_script_has_todo(self):
        script = _generate_reproduction_script("something broke", BugType.UNKNOWN)
        self.assertIn("TODO", script)


if __name__ == "__main__":
    unittest.main()
