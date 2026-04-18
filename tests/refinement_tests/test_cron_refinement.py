"""
Refinement test: Abstract Cron State Machine vs Concrete Implementation.

Checks that the concrete cron job operations (from SM-1 spec and
thrice/invariants/cron_invariants.py) refine the abstract state machine.

Refinement means:
  1. Every concrete state maps to an abstract state
  2. Every concrete transition is allowed by the abstract transition table
  3. Abstract invariants hold on inferred concrete states
  4. State inference is total (every valid concrete config maps to a state)

This is the "simulation relation" between the TLA+ spec and Python impl.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "thrice"))

pytest.importorskip(
    "invariants.cron_invariants",
    reason="hermes-agent invariants.cron_invariants not available",
)

from invariants.cron_invariants import (  # noqa: E402
    _VALID_CRON_TRANSITIONS,
    CRON_STATES,
    infer_cron_state,
)

pytestmark = [pytest.mark.refinement, pytest.mark.requires_hermes]


# -- Helpers ------------------------------------------------------------------

def _past_iso(minutes=60):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

def _future_iso(minutes=60):
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


# ============================================================================
# Abstract state machine definition (from SM-1 spec)
# ============================================================================

ABSTRACT_STATES = {"nonexistent", "scheduled", "running", "paused", "completed", "failed", "removed"}

ABSTRACT_TRANSITIONS = {
    "nonexistent": {"scheduled"},
    "scheduled":   {"running", "paused", "removed"},
    "running":     {"scheduled", "completed", "failed"},
    "paused":      {"scheduled", "removed"},
    "completed":   {"scheduled", "paused", "removed"},  # rescheduled or disabled after completion
    "failed":      {"scheduled", "paused", "removed"},
    "removed":     set(),
}

# Mapping from concrete (infer_cron_state) states to abstract states
CONCRETE_TO_ABSTRACT = {
    "created":    "scheduled",     # created maps to scheduled (initial state)
    "scheduled":  "scheduled",
    "running":    "running",
    "due":        "scheduled",     # due is a sub-state of scheduled
    "disabled":   "paused",        # disabled maps to paused
    "completed":  "completed",
    "failed":     "failed",
    "deleted":    "removed",
}


# ============================================================================
# Test 1: State inference is total
# ============================================================================

class TestStateInferenceTotal:
    """Every valid combination of concrete fields maps to a known state."""

    def test_all_enabled_status_combos(self):
        """Exhaustively test enabled x last_status x next_run_at x is_running."""
        for enabled in [True, False]:
            for last_status in [None, "ok", "error"]:
                for has_next_run in [True, False]:
                    for is_running in [True, False]:
                        job = {
                            "enabled": enabled,
                            "last_status": last_status,
                            "next_run_at": _future_iso() if has_next_run else None,
                            "last_run_at": _past_iso() if last_status else None,
                            "last_error": "err" if last_status == "error" else None,
                        }
                        state = infer_cron_state(job, is_running=is_running)
                        assert state in CRON_STATES, (
                            f"Inferred state '{state}' not in CRON_STATES for "
                            f"enabled={enabled}, last_status={last_status}, "
                            f"has_next_run={has_next_run}, is_running={is_running}"
                        )

    def test_inference_deterministic(self):
        """Same input always produces same output."""
        job = {"enabled": True, "last_status": None, "next_run_at": _future_iso()}
        s1 = infer_cron_state(job)
        s2 = infer_cron_state(job)
        assert s1 == s2


# ============================================================================
# Test 2: Concrete states map to abstract states
# ============================================================================

class TestConcreteToAbstractMapping:
    """Every concrete state has a mapping to an abstract state."""

    def test_all_concrete_states_mapped(self):
        for concrete_state in CRON_STATES:
            assert concrete_state in CONCRETE_TO_ABSTRACT, (
                f"Concrete state '{concrete_state}' has no abstract mapping"
            )

    def test_abstract_states_are_valid(self):
        for concrete, abstract in CONCRETE_TO_ABSTRACT.items():
            assert abstract in ABSTRACT_STATES, (
                f"Concrete '{concrete}' maps to invalid abstract '{abstract}'"
            )


# ============================================================================
# Test 3: Concrete transitions refine abstract transitions
# ============================================================================

class TestTransitionRefinement:
    """Every valid concrete transition maps to a valid abstract transition."""

    def test_all_concrete_transitions_valid_in_abstract(self):
        """For every (from, to) in concrete transitions, the corresponding
        abstract (from, to) must be in ABSTRACT_TRANSITIONS."""
        for from_concrete, to_set in _VALID_CRON_TRANSITIONS.items():
            from_abstract = CONCRETE_TO_ABSTRACT.get(from_concrete)
            if from_abstract is None:
                continue  # skip unmapped states

            for to_concrete in to_set:
                to_abstract = CONCRETE_TO_ABSTRACT.get(to_concrete)
                if to_abstract is None:
                    continue

                # Self-transitions in abstract are always valid
                if from_abstract == to_abstract:
                    continue

                allowed = ABSTRACT_TRANSITIONS.get(from_abstract, set())
                assert to_abstract in allowed, (
                    f"Concrete transition {from_concrete}->{to_concrete} "
                    f"maps to abstract {from_abstract}->{to_abstract} "
                    f"which is NOT in abstract transition table. "
                    f"Allowed from {from_abstract}: {allowed}"
                )

    def test_no_concrete_transition_violates_abstract_terminal(self):
        """Concrete transitions from terminal abstract states
        must not exist (except self-loops)."""
        terminal_abstract = {"removed"}
        terminal_concrete = {
            c for c, a in CONCRETE_TO_ABSTRACT.items()
            if a in terminal_abstract
        }
        for tc in terminal_concrete:
            targets = _VALID_CRON_TRANSITIONS.get(tc, set())
            for target in targets:
                target_abstract = CONCRETE_TO_ABSTRACT.get(target)
                if target_abstract and target_abstract != CONCRETE_TO_ABSTRACT.get(tc):
                    pytest.fail(
                        f"Concrete state '{tc}' (abstract: removed) "
                        f"has transition to '{target}' (abstract: {target_abstract})"
                    )


# ============================================================================
# Test 4: Invariant preservation under refinement
# ============================================================================

class TestInvariantPreservation:
    """Abstract invariants still hold when checked on concrete states."""

    def test_paused_implies_disabled(self):
        """SM-1 INV-C1: paused => enabled=false.
        In concrete: disabled state => enabled=false (by construction)."""
        job_disabled = {"enabled": False, "last_status": None, "next_run_at": None}
        state = infer_cron_state(job_disabled)
        assert state == "disabled"
        # disabled maps to abstract "paused"
        assert CONCRETE_TO_ABSTRACT[state] == "paused"
        # And enabled is False
        assert not job_disabled["enabled"]

    def test_completed_transitions_limited(self):
        """SM-1: completed can go to scheduled, paused, or removed."""
        abstract_targets = ABSTRACT_TRANSITIONS["completed"]
        assert abstract_targets == {"scheduled", "paused", "removed"}

    def test_running_not_removable(self):
        """SM-1 INV-C2: running -> removed not in abstract transitions."""
        assert "removed" not in ABSTRACT_TRANSITIONS["running"]

    def test_concrete_scheduled_refines_abstract_scheduled(self):
        """A concrete 'scheduled' job with valid fields refines
        the abstract 'scheduled' state."""
        job = {
            "enabled": True,
            "last_status": None,
            "next_run_at": _future_iso(),
        }
        state = infer_cron_state(job)
        assert state == "scheduled"
        assert CONCRETE_TO_ABSTRACT[state] == "scheduled"


# ============================================================================
# Test 5: Operation simulation
# ============================================================================

class TestOperationSimulation:
    """Simulate abstract operations and verify concrete checker agrees."""

    def test_create_operation(self):
        """Abstract: nonexistent -> scheduled (via create).
        Concrete: a new job should infer as 'scheduled'."""
        job = {
            "enabled": True,
            "last_status": None,
            "next_run_at": _future_iso(),
            "last_run_at": None,
        }
        state = infer_cron_state(job)
        assert CONCRETE_TO_ABSTRACT[state] == "scheduled"

    def test_tick_operation(self):
        """Abstract: scheduled -> running (via tick).
        Concrete: is_running=True should infer as 'running'."""
        job = {"enabled": True, "last_status": None, "next_run_at": _future_iso()}
        state = infer_cron_state(job, is_running=True)
        assert CONCRETE_TO_ABSTRACT[state] == "running"

    def test_pause_operation(self):
        """Abstract: scheduled -> paused (via pause).
        Concrete: enabled=False should infer as 'disabled' (maps to paused)."""
        job = {"enabled": False, "last_status": None, "next_run_at": None}
        state = infer_cron_state(job)
        assert CONCRETE_TO_ABSTRACT[state] == "paused"

    def test_success_recurring_operation(self):
        """Abstract: running -> scheduled (recurring success).
        Concrete: after success, enabled + next_run -> scheduled."""
        job = {
            "enabled": True,
            "last_status": "ok",
            "next_run_at": _future_iso(),
            "last_run_at": _past_iso(),
        }
        state = infer_cron_state(job, is_running=False)
        assert CONCRETE_TO_ABSTRACT[state] == "scheduled"

    def test_success_oneshot_operation(self):
        """Abstract: running -> completed (one-shot success).
        Concrete: after success, no next_run -> completed."""
        job = {
            "enabled": True,
            "last_status": "ok",
            "next_run_at": None,
            "last_run_at": _past_iso(),
        }
        state = infer_cron_state(job, is_running=False)
        assert CONCRETE_TO_ABSTRACT[state] == "completed"

    def test_failure_operation(self):
        """Abstract: running -> failed.
        Concrete: last_status=error, no next_run -> failed."""
        job = {
            "enabled": True,
            "last_status": "error",
            "last_error": "timeout",
            "next_run_at": None,
            "last_run_at": _past_iso(),
        }
        state = infer_cron_state(job, is_running=False)
        assert CONCRETE_TO_ABSTRACT[state] == "failed"


# ============================================================================
# Test 6: Transition table completeness
# ============================================================================

class TestTransitionTableCompleteness:
    """Both abstract and concrete transition tables are well-formed."""

    def test_abstract_covers_all_states(self):
        for state in ABSTRACT_STATES:
            assert state in ABSTRACT_TRANSITIONS, f"Abstract state {state} missing"

    def test_concrete_covers_all_states(self):
        for state in CRON_STATES:
            assert state in _VALID_CRON_TRANSITIONS, f"Concrete state {state} missing"

    def test_abstract_targets_are_valid(self):
        for src, targets in ABSTRACT_TRANSITIONS.items():
            for tgt in targets:
                assert tgt in ABSTRACT_STATES, f"Invalid abstract target {tgt}"

    def test_concrete_targets_are_valid(self):
        for src, targets in _VALID_CRON_TRANSITIONS.items():
            for tgt in targets:
                assert tgt in CRON_STATES, f"Invalid concrete target {tgt}"
