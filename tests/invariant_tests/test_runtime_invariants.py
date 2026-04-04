"""
Unit tests for hermes_invariants.py (new-files/hermes_invariants.py).

Tests every public method of InvariantChecker with valid and invalid inputs.
"""

import os
import sys
import types
import pytest

# Ensure hermes_invariants is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "new-files"))

from hermes_invariants import InvariantChecker

pytestmark = pytest.mark.invariant


# ============================================================================
# check_message_integrity
# ============================================================================

class TestCheckMessageIntegrity:
    def test_empty_messages(self):
        violations = InvariantChecker.check_message_integrity([])
        assert any("EMPTY_MESSAGES" in v for v in violations)

    def test_valid_conversation(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert violations == []

    def test_invalid_role(self):
        msgs = [{"role": "oracle", "content": "x"}]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert any("INVALID_ROLE" in v for v in violations)

    def test_not_a_dict_message(self):
        msgs = ["not a dict"]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert any("INVALID_MESSAGE" in v for v in violations)

    def test_orphan_tool_result_after_assistant_no_calls(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Checking"},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert any("ORPHAN_TOOL_RESULT" in v for v in violations)

    def test_orphan_tool_result_after_user(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert any("ORPHAN_TOOL_RESULT" in v for v in violations)

    def test_valid_tool_result_after_assistant_with_calls(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert violations == []

    def test_consecutive_tool_results_valid(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert violations == []

    def test_valid_roles_only(self):
        for role in ["system", "user", "assistant", "tool"]:
            msgs = [{"role": role, "content": "x"}]
            if role == "tool":
                msgs[0]["tool_call_id"] = "tc"
            violations = InvariantChecker.check_message_integrity(msgs)
            assert not any("INVALID_ROLE" in v for v in violations)

    def test_tool_after_tool_valid(self):
        """Multiple tool results in a row (following an assistant with calls)."""
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "tool", "tool_call_id": "c3", "content": "r3"},
        ]
        violations = InvariantChecker.check_message_integrity(msgs)
        assert violations == []


# ============================================================================
# check_cron_invariants
# ============================================================================

class TestCheckCronInvariants:
    def test_valid_jobs(self):
        jobs = [
            {"id": "j1", "state": "scheduled", "enabled": True, "next_run_at": 9999999999},
            {"id": "j2", "state": "paused", "enabled": False},
        ]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert violations == []

    def test_duplicate_ids(self):
        jobs = [
            {"id": "j1", "state": "scheduled"},
            {"id": "j1", "state": "paused"},
        ]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert any("DUPLICATE_JOB_ID" in v for v in violations)

    def test_invalid_state(self):
        jobs = [{"id": "j1", "state": "banana_state"}]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert any("INVALID_CRON_STATE" in v for v in violations)

    def test_completed_with_next_run(self):
        jobs = [{"id": "j1", "state": "completed", "next_run_at": 12345}]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert any("COMPLETED_WITH_NEXT_RUN" in v for v in violations)

    def test_repeat_overflow(self):
        jobs = [{"id": "j1", "state": "scheduled",
                 "repeat": {"times": 3, "completed": 5}}]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert any("REPEAT_OVERFLOW" in v for v in violations)

    def test_empty_list_passes(self):
        violations = InvariantChecker.check_cron_invariants([])
        assert violations == []

    def test_repeat_under_limit(self):
        jobs = [{"id": "j1", "state": "scheduled",
                 "repeat": {"times": 10, "completed": 5}}]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert not any("REPEAT_OVERFLOW" in v for v in violations)

    def test_repeat_at_limit(self):
        jobs = [{"id": "j1", "state": "scheduled",
                 "repeat": {"times": 5, "completed": 5}}]
        violations = InvariantChecker.check_cron_invariants(jobs)
        assert not any("REPEAT_OVERFLOW" in v for v in violations)


# ============================================================================
# check_process_registry
# ============================================================================

class TestCheckProcessRegistry:
    class FakeSession:
        def __init__(self, exited=False):
            self.exited = exited

    def test_clean_registry(self):
        running = {"p1": self.FakeSession(exited=False)}
        finished = {"p2": self.FakeSession(exited=True)}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert violations == []

    def test_dual_registry(self):
        s = self.FakeSession(exited=True)
        running = {"p1": s}
        finished = {"p1": s}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert any("DUAL_REGISTRY" in v for v in violations)

    def test_running_but_exited(self):
        running = {"p1": self.FakeSession(exited=True)}
        finished = {}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert any("RUNNING_BUT_EXITED" in v for v in violations)

    def test_finished_not_exited(self):
        running = {}
        finished = {"p1": self.FakeSession(exited=False)}
        violations = InvariantChecker.check_process_registry(running, finished)
        assert any("FINISHED_NOT_EXITED" in v for v in violations)

    def test_empty_registry(self):
        violations = InvariantChecker.check_process_registry({}, {})
        assert violations == []

    def test_multiple_running_ok(self):
        running = {
            "p1": self.FakeSession(exited=False),
            "p2": self.FakeSession(exited=False),
        }
        violations = InvariantChecker.check_process_registry(running, {})
        assert violations == []


# ============================================================================
# check_session_invariants
# ============================================================================

class TestCheckSessionInvariants:
    class FakeEntry:
        def __init__(self, session_id, input_tokens=0, output_tokens=0, total_tokens=0):
            self.session_id = session_id
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.total_tokens = total_tokens

    def test_valid_sessions(self):
        entries = {"k1": self.FakeEntry("s1"), "k2": self.FakeEntry("s2")}
        violations = InvariantChecker.check_session_invariants(entries)
        assert violations == []

    def test_missing_session_id(self):
        class NoId:
            session_id = None
        entries = {"k1": NoId()}
        violations = InvariantChecker.check_session_invariants(entries)
        assert any("MISSING_SESSION_ID" in v for v in violations)

    def test_duplicate_session_id(self):
        entries = {"k1": self.FakeEntry("s1"), "k2": self.FakeEntry("s1")}
        violations = InvariantChecker.check_session_invariants(entries)
        assert any("DUPLICATE_SESSION_ID" in v for v in violations)

    def test_negative_tokens(self):
        entries = {"k1": self.FakeEntry("s1", input_tokens=-5)}
        violations = InvariantChecker.check_session_invariants(entries)
        assert any("NEGATIVE_TOKENS" in v for v in violations)

    def test_zero_tokens_ok(self):
        entries = {"k1": self.FakeEntry("s1", input_tokens=0, output_tokens=0)}
        violations = InvariantChecker.check_session_invariants(entries)
        assert violations == []


# ============================================================================
# check_delegation
# ============================================================================

class TestCheckDelegation:
    def test_valid_delegation(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[1, 2], depth=1, max_depth=2
        )
        assert violations == []

    def test_depth_exceeded(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[], depth=3, max_depth=2
        )
        assert any("DEPTH_EXCEEDED" in v for v in violations)

    def test_too_many_children(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[1, 2, 3, 4], depth=0, max_depth=5
        )
        assert any("TOO_MANY_CHILDREN" in v for v in violations)

    def test_duplicate_child(self):
        child = object()
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[child, child], depth=0, max_depth=5
        )
        assert any("DUPLICATE_CHILD" in v for v in violations)

    def test_at_max_depth_ok(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[], depth=2, max_depth=2
        )
        assert not any("DEPTH_EXCEEDED" in v for v in violations)

    def test_three_children_ok(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[1, 2, 3], depth=0, max_depth=5
        )
        assert not any("TOO_MANY_CHILDREN" in v for v in violations)

    def test_no_children_ok(self):
        violations = InvariantChecker.check_delegation_tree(
            parent_agent=None, active_children=[], depth=0, max_depth=2
        )
        assert violations == []
