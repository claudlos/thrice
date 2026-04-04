"""
Tests for silent_error_audit — Thrice Improvement #12

30+ tests covering SilentErrorScanner, HealthDashboard, LoggingFixer, and helpers.
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import TestCase, main

# Ensure the module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "new-files"))

from silent_error_audit import (
    HealthDashboard,
    LoggingFixer,
    ModuleHealth,
    SilentErrorPattern,
    SilentErrorScanner,
    quick_health_check,
)


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------

def _write_tmp(code: str, suffix: str = ".py") -> str:
    """Write code to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(code))
    return path


# =========================================================================
# SilentErrorPattern dataclass
# =========================================================================

class TestSilentErrorPattern(TestCase):

    def test_fields_exist(self):
        p = SilentErrorPattern("a.py", 1, "bare_except", "except:", "fix it")
        self.assertEqual(p.file_path, "a.py")
        self.assertEqual(p.line_number, 1)
        self.assertEqual(p.pattern_type, "bare_except")
        self.assertEqual(p.code_snippet, "except:")
        self.assertEqual(p.suggestion, "fix it")

    def test_equality(self):
        a = SilentErrorPattern("a.py", 1, "bare_except", "x", "y")
        b = SilentErrorPattern("a.py", 1, "bare_except", "x", "y")
        self.assertEqual(a, b)


# =========================================================================
# SilentErrorScanner
# =========================================================================

class TestSilentErrorScanner(TestCase):

    def setUp(self):
        self.scanner = SilentErrorScanner()

    # --- bare except ---

    def test_detect_bare_except_pass(self):
        code = """\
        try:
            x = 1
        except:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].pattern_type, "bare_except")
        finally:
            os.unlink(path)

    def test_detect_bare_except_ellipsis(self):
        code = """\
        try:
            x = 1
        except:
            ...
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].pattern_type, "bare_except")
        finally:
            os.unlink(path)

    # --- broad except pass ---

    def test_detect_broad_except_pass(self):
        code = """\
        try:
            x = 1
        except Exception:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].pattern_type, "broad_except_pass")
        finally:
            os.unlink(path)

    def test_detect_broad_except_as_pass(self):
        code = """\
        try:
            x = 1
        except Exception as e:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].pattern_type, "broad_except_pass")
        finally:
            os.unlink(path)

    def test_detect_base_exception_pass(self):
        code = """\
        try:
            x = 1
        except BaseException:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertIn(results[0].pattern_type, ("broad_except_pass",))
        finally:
            os.unlink(path)

    # --- specific except pass ---

    def test_detect_specific_except_pass(self):
        code = """\
        try:
            x = int("foo")
        except ValueError:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].pattern_type, "except_pass")
        finally:
            os.unlink(path)

    # --- no false positives ---

    def test_no_false_positive_with_logging(self):
        code = """\
        import logging
        logger = logging.getLogger(__name__)
        try:
            x = 1
        except Exception as e:
            logger.exception("oops: %s", e)
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)

    def test_no_false_positive_with_print(self):
        code = """\
        try:
            x = 1
        except Exception as e:
            print(f"error: {e}")
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)

    def test_no_false_positive_with_raise(self):
        code = """\
        try:
            x = 1
        except Exception:
            raise
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)

    # --- multiple patterns ---

    def test_multiple_patterns_in_one_file(self):
        code = """\
        try:
            a = 1
        except:
            pass

        try:
            b = 2
        except Exception:
            pass

        try:
            c = 3
        except ValueError:
            pass
        """
        path = _write_tmp(code)
        try:
            results = self.scanner.scan_file(path)
            self.assertTrue(len(results) >= 3)
        finally:
            os.unlink(path)

    # --- file handling ---

    def test_nonexistent_file(self):
        results = self.scanner.scan_file("/nonexistent/path.py")
        self.assertEqual(results, [])

    def test_empty_file(self):
        path = _write_tmp("")
        try:
            results = self.scanner.scan_file(path)
            self.assertEqual(results, [])
        finally:
            os.unlink(path)

    # --- directory scanning ---

    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "a.py")
            with open(p1, "w") as f:
                f.write("try:\n    x=1\nexcept:\n    pass\n")
            p2 = os.path.join(td, "b.py")
            with open(p2, "w") as f:
                f.write("x = 1\n")
            results = self.scanner.scan_directory(td)
            self.assertTrue(len(results) >= 1)

    def test_scan_directory_nonexistent(self):
        results = self.scanner.scan_directory("/no/such/dir")
        self.assertEqual(results, [])

    def test_scan_directory_glob(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "a.py")
            with open(p1, "w") as f:
                f.write("try:\n    x=1\nexcept:\n    pass\n")
            p2 = os.path.join(td, "b.txt")
            with open(p2, "w") as f:
                f.write("try:\n    x=1\nexcept:\n    pass\n")
            results = self.scanner.scan_directory(td, "*.py")
            # Only .py files
            for r in results:
                self.assertTrue(r.file_path.endswith(".py"))

    # --- report ---

    def test_generate_report_empty(self):
        report = SilentErrorScanner.generate_report([])
        self.assertIn("clean", report.lower())

    def test_generate_report_with_patterns(self):
        patterns = [
            SilentErrorPattern("a.py", 3, "bare_except", "except:", "fix"),
            SilentErrorPattern("b.py", 10, "except_pass", "except V:", "fix"),
        ]
        report = SilentErrorScanner.generate_report(patterns)
        self.assertIn("2 pattern(s)", report)
        self.assertIn("bare_except", report)
        self.assertIn("except_pass", report)


# =========================================================================
# ModuleHealth dataclass
# =========================================================================

class TestModuleHealth(TestCase):

    def test_defaults(self):
        m = ModuleHealth(module_name="foo", importable=True)
        self.assertIsNone(m.import_error)
        self.assertIsNone(m.version)
        self.assertFalse(m.has_tests)
        self.assertEqual(m.test_count, 0)

    def test_fields(self):
        m = ModuleHealth("bar", False, "ImportError: x", "1.0", True, 5)
        self.assertEqual(m.module_name, "bar")
        self.assertFalse(m.importable)
        self.assertEqual(m.import_error, "ImportError: x")
        self.assertEqual(m.version, "1.0")
        self.assertTrue(m.has_tests)
        self.assertEqual(m.test_count, 5)


# =========================================================================
# HealthDashboard
# =========================================================================

class TestHealthDashboard(TestCase):

    def test_thrice_modules_list(self):
        self.assertIn("silent_error_audit", HealthDashboard.THRICE_MODULES)
        self.assertTrue(len(HealthDashboard.THRICE_MODULES) >= 29)

    def test_check_module_importable(self):
        # 'os' should always be importable
        dash = HealthDashboard(".")
        health = dash.check_module("os")
        self.assertTrue(health.importable)
        self.assertIsNone(health.import_error)

    def test_check_module_not_importable(self):
        dash = HealthDashboard(".")
        health = dash.check_module("nonexistent_module_xyz_999")
        self.assertFalse(health.importable)
        self.assertIsNotNone(health.import_error)

    def test_check_all_returns_list(self):
        dash = HealthDashboard(".")
        results = dash.check_all_thrice_modules()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(HealthDashboard.THRICE_MODULES))

    def test_get_active_modules(self):
        dash = HealthDashboard(".")
        active = dash.get_active_modules()
        self.assertIsInstance(active, list)

    def test_get_failed_modules(self):
        dash = HealthDashboard(".")
        failed = dash.get_failed_modules()
        self.assertIsInstance(failed, list)

    def test_format_dashboard(self):
        dash = HealthDashboard(".")
        output = dash.format_dashboard()
        self.assertIn("Thrice Module Health Dashboard", output)
        self.assertIn("Total modules", output)

    def test_check_module_with_tests(self):
        with tempfile.TemporaryDirectory() as td:
            tests_dir = os.path.join(td, "tests")
            os.makedirs(tests_dir)
            test_file = os.path.join(tests_dir, "test_os.py")
            with open(test_file, "w") as f:
                f.write("def test_one(): pass\ndef test_two(): pass\n")
            dash = HealthDashboard(td)
            health = dash.check_module("os")
            self.assertTrue(health.has_tests)
            self.assertEqual(health.test_count, 2)


# =========================================================================
# LoggingFixer
# =========================================================================

class TestLoggingFixer(TestCase):

    def setUp(self):
        self.fixer = LoggingFixer()

    def test_suggest_fix_bare_except(self):
        p = SilentErrorPattern("a.py", 3, "bare_except", "    except:", "")
        fix = self.fixer.suggest_fix(p)
        self.assertIn("except Exception as exc", fix)
        self.assertIn("logger.exception", fix)
        self.assertIn("raise", fix)

    def test_suggest_fix_broad_except_pass(self):
        p = SilentErrorPattern("a.py", 3, "broad_except_pass", "    except Exception:", "")
        fix = self.fixer.suggest_fix(p)
        self.assertIn("except Exception as exc", fix)
        self.assertIn("logger.exception", fix)

    def test_suggest_fix_except_pass(self):
        p = SilentErrorPattern("a.py", 3, "except_pass", "    except ValueError:", "")
        fix = self.fixer.suggest_fix(p)
        self.assertIn("ValueError", fix)
        self.assertIn("logger.debug", fix)

    def test_suggest_fix_broad_no_log(self):
        p = SilentErrorPattern("a.py", 3, "broad_except_no_log", "    except Exception:", "")
        fix = self.fixer.suggest_fix(p)
        self.assertIn("TODO", fix)

    def test_fix_pattern_applies_to_code(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        p = SilentErrorPattern("a.py", 3, "bare_except", "except:", "fix")
        result = self.fixer.fix_pattern(code, p)
        self.assertIn("except Exception as exc", result)
        self.assertNotIn("except:\n    pass", result)

    def test_generate_patch_empty(self):
        self.assertEqual(self.fixer.generate_patch([]), "")

    def test_generate_patch_with_file(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        path = _write_tmp(code)
        try:
            p = SilentErrorPattern(path, 3, "bare_except", "except:", "fix")
            patch = self.fixer.generate_patch([p])
            self.assertIn("---", patch)
            self.assertIn("+++", patch)
        finally:
            os.unlink(path)

    def test_fix_preserves_indent(self):
        p = SilentErrorPattern("a.py", 1, "bare_except", "        except:", "")
        fix = self.fixer.suggest_fix(p)
        self.assertTrue(fix.startswith("        "))

    def test_fix_pattern_out_of_range(self):
        code = "x = 1\n"
        p = SilentErrorPattern("a.py", 99, "bare_except", "except:", "fix")
        result = self.fixer.fix_pattern(code, p)
        self.assertEqual(result, code)


# =========================================================================
# quick_health_check
# =========================================================================

class TestQuickHealthCheck(TestCase):

    def test_returns_string(self):
        result = quick_health_check(".")
        self.assertIsInstance(result, str)

    def test_contains_summary(self):
        result = quick_health_check(".")
        self.assertIn("Thrice health:", result)
        self.assertIn("importable", result)


# =========================================================================
# Integration / edge cases
# =========================================================================

class TestIntegration(TestCase):

    def test_scan_and_fix_roundtrip(self):
        code = textwrap.dedent("""\
        try:
            x = 1
        except:
            pass
        """)
        path = _write_tmp(code)
        try:
            scanner = SilentErrorScanner()
            patterns = scanner.scan_file(path)
            self.assertTrue(len(patterns) >= 1)

            fixer = LoggingFixer()
            fixed = fixer.fix_pattern(code, patterns[0])
            self.assertIn("logger", fixed)
            self.assertNotIn("except:\n    pass", fixed)
        finally:
            os.unlink(path)

    def test_syntax_error_file_does_not_crash(self):
        code = "def foo(\n"  # intentional syntax error
        path = _write_tmp(code)
        try:
            scanner = SilentErrorScanner()
            # Should not raise
            results = scanner.scan_file(path)
            self.assertIsInstance(results, list)
        finally:
            os.unlink(path)

    def test_nested_try_except(self):
        code = textwrap.dedent("""\
        try:
            try:
                x = 1
            except:
                pass
        except Exception:
            pass
        """)
        path = _write_tmp(code)
        try:
            scanner = SilentErrorScanner()
            results = scanner.scan_file(path)
            self.assertTrue(len(results) >= 2)
        finally:
            os.unlink(path)

    def test_broad_except_no_logging(self):
        code = textwrap.dedent("""\
        try:
            x = 1
        except Exception as e:
            result = None
            x = 2
        """)
        path = _write_tmp(code)
        try:
            scanner = SilentErrorScanner()
            results = scanner.scan_file(path)
            types = [r.pattern_type for r in results]
            self.assertIn("broad_except_no_log", types)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    main()
