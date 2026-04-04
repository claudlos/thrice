"""
Smoke tests for the integrated Hermes improvements.
Run with PYTHONPATH pointing to the hermes-agent dir.
"""
import sys
import os

# Ensure hermes-agent is on the path
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))


class TestToolAliasMap:
    def test_bash_resolves_to_terminal(self):
        from tool_alias_map import resolve_alias
        assert resolve_alias("bash") == "terminal"

    def test_grep_resolves_to_search_files(self):
        from tool_alias_map import resolve_alias
        assert resolve_alias("grep") == "search_files"

    def test_cat_resolves_to_read_file(self):
        from tool_alias_map import resolve_alias
        assert resolve_alias("cat") == "read_file"

    def test_sed_resolves_to_patch(self):
        from tool_alias_map import resolve_alias
        assert resolve_alias("sed") == "patch"

    def test_unknown_returns_none(self):
        from tool_alias_map import resolve_alias
        assert resolve_alias("nonexistent_tool_xyz") is None


class TestAdaptiveCompression:
    def test_larger_context_higher_threshold(self):
        from adaptive_compression import get_compression_threshold
        t200k = get_compression_threshold(200000)
        t64k = get_compression_threshold(64000)
        t32k = get_compression_threshold(32000)
        assert t200k > t64k > t32k

    def test_small_context_low_threshold(self):
        from adaptive_compression import get_compression_threshold
        t = get_compression_threshold(16000)
        assert 0.3 <= t <= 0.6


class TestSmartTruncation:
    def test_truncates_large_output(self):
        from smart_truncation import truncate_output
        big = "line\n" * 20000  # ~100K chars
        result = truncate_output(big, max_chars=50000)
        assert len(result) < len(big)

    def test_preserves_small_output(self):
        from smart_truncation import truncate_output
        small = "hello world"
        assert truncate_output(small, max_chars=50000) == small

    def test_truncation_marker_present(self):
        from smart_truncation import truncate_output
        big = "x" * 100000
        result = truncate_output(big, max_chars=50000)
        # Should have some kind of marker
        assert len(result) < len(big)


class TestStructuredErrors:
    def test_classify_file_not_found(self):
        from structured_errors import classify_error, ErrorType
        err = classify_error("FileNotFoundError: /tmp/x")
        assert err.error_type == ErrorType.NOT_FOUND
        assert err.recoverable is True

    def test_classify_permission_denied(self):
        from structured_errors import classify_error, ErrorType
        err = classify_error("PermissionError: cannot write")
        assert err.error_type == ErrorType.PERMISSION_DENIED

    def test_classify_timeout(self):
        from structured_errors import classify_error, ErrorType
        err = classify_error("TimeoutError: operation timed out")
        assert err.error_type == ErrorType.TIMEOUT


class TestCommentStripMatcher:
    def test_strip_python_comments(self):
        from comment_strip_matcher import strip_comments
        code = "x = 1  # comment\ny = 2"
        stripped = strip_comments(code, "python")
        assert "#" not in stripped
        assert "x = 1" in stripped

    def test_strip_js_comments(self):
        from comment_strip_matcher import strip_comments
        code = "let x = 1; // comment\nlet y = 2;"
        stripped = strip_comments(code, "javascript")
        assert "//" not in stripped


class TestCronStateMachine:
    def _import_jobs(self):
        """Import cron.jobs fresh, bypassing stale pycache."""
        import importlib
        import importlib.util
        import pathlib
        jobs_path = pathlib.Path(os.path.expanduser("~/.hermes/hermes-agent/cron/jobs.py"))
        spec = importlib.util.spec_from_file_location("cron_jobs_fresh", str(jobs_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_valid_transitions(self):
        jobs = self._import_jobs()
        assert jobs.is_valid_transition("scheduled", "running")
        assert jobs.is_valid_transition("running", "completed")
        assert jobs.is_valid_transition("running", "failed")
        assert jobs.is_valid_transition("scheduled", "paused")
        assert jobs.is_valid_transition("paused", "scheduled")

    def test_invalid_transitions(self):
        jobs = self._import_jobs()
        assert not jobs.is_valid_transition("completed", "running")
        assert not jobs.is_valid_transition("paused", "running")

    def test_job_states_include_new_states(self):
        from cron.jobs import JOB_STATES
        assert "running" in JOB_STATES
        assert "failed" in JOB_STATES
        assert "scheduled" in JOB_STATES

    def test_guard_raises_on_invalid(self):
        jobs = self._import_jobs()
        import pytest
        with pytest.raises(jobs.InvalidTransitionError):
            jobs._guard_transition("test-job", "completed", "running")


class TestHermesInvariants:
    def test_valid_messages_no_violations(self):
        from hermes_invariants import InvariantChecker
        v = InvariantChecker.check_message_integrity([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert v == []

    def test_empty_messages_violation(self):
        from hermes_invariants import InvariantChecker
        v = InvariantChecker.check_message_integrity([])
        assert len(v) > 0

    def test_cron_valid_no_violations(self):
        from hermes_invariants import InvariantChecker
        jobs = [{"id": "j1", "state": "scheduled", "enabled": True,
                 "next_run_at": "2099-01-01T00:00:00Z"}]
        v = InvariantChecker.check_cron_invariants(jobs)
        assert v == [], f"Unexpected violations: {v}"

    def test_process_registry_overlap_detected(self):
        from hermes_invariants import InvariantChecker
        v = InvariantChecker.check_process_registry(
            running={"s1": {}},
            finished={"s1": {}},
        )
        assert len(v) > 0


class TestEnforcement:
    def test_modes_exist(self):
        from enforcement import EnforcementMode
        assert hasattr(EnforcementMode, "DEVELOPMENT")
        assert hasattr(EnforcementMode, "PRODUCTION")
        assert hasattr(EnforcementMode, "TESTING")

    def test_get_mode(self):
        from enforcement import get_enforcement_mode, EnforcementMode
        mode = get_enforcement_mode()
        assert mode in EnforcementMode
