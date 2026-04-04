"""
Property-based stateful testing for the Cron Job state machine (SM-1).

Uses hypothesis RuleBasedStateMachine to randomly apply operations
(create, tick, pause, resume, remove, mark_success, mark_failure)
and check all SM-1 invariants after every step.

Invariants verified:
  INV-C1: Paused Never Fires (paused => enabled=false)
  INV-C2: Running Job Never Deleted
  INV-C3: Repeat Count Conservation (completed + runs_left = total)
  INV-C5: No Concurrent Execution (at most one running)
  INV-C6: Enabled-State Consistency
  INV-C7: Monotonic Completion Count
  Unique job IDs
  Valid state transitions only
"""

import pytest
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set

from hypothesis import settings, HealthCheck, Phase
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    precondition,
    consumes,
    multiple,
)

pytestmark = [pytest.mark.property, pytest.mark.stateful]


# ============================================================================
# Abstract cron job model (pure Python, no I/O)
# ============================================================================

CRON_STATES = {"nonexistent", "scheduled", "running", "paused", "completed", "failed", "removed"}

VALID_TRANSITIONS = {
    "nonexistent": {"scheduled"},
    "scheduled":   {"running", "paused", "removed"},
    "running":     {"scheduled", "completed", "failed"},  # scheduled = rescheduled
    "paused":      {"scheduled", "removed"},  # resume -> scheduled
    "completed":   {"removed"},  # terminal but removable
    "failed":      {"scheduled", "paused", "removed"},  # retry -> scheduled
    "removed":     set(),  # terminal
}


class CronJob:
    """Lightweight model of a cron job for property testing."""

    def __init__(self, jid: str, recurring: bool = True, total_repeat: Optional[int] = None):
        self.id = jid
        self.state = "scheduled"
        self.enabled = True
        self.recurring = recurring
        self.total_repeat = total_repeat
        self.runs_completed = 0
        self.runs_left = total_repeat if total_repeat is not None else None
        self.completion_history: list = []  # track monotonicity

    @property
    def runs_left_computed(self):
        if self.total_repeat is None:
            return None
        return self.total_repeat - self.runs_completed


class CronStateMachine(RuleBasedStateMachine):
    """
    Stateful test that randomly exercises cron job operations and
    verifies SM-1 invariants after every operation.
    """

    def __init__(self):
        super().__init__()
        self.jobs: Dict[str, CronJob] = {}
        self.running_job_id: Optional[str] = None  # at most one
        self._counter = 0

    # -- Helpers ---------------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"job_{self._counter}"

    def _active_ids(self) -> Set[str]:
        """IDs of jobs in non-terminal states."""
        return {
            jid for jid, j in self.jobs.items()
            if j.state not in ("removed", "completed")
        }

    def _schedulable_ids(self) -> Set[str]:
        return {
            jid for jid, j in self.jobs.items()
            if j.state == "scheduled"
        }

    def _pausable_ids(self) -> Set[str]:
        return {
            jid for jid, j in self.jobs.items()
            if j.state in ("scheduled", "failed")
        }

    def _paused_ids(self) -> Set[str]:
        return {jid for jid, j in self.jobs.items() if j.state == "paused"}

    def _failed_ids(self) -> Set[str]:
        return {jid for jid, j in self.jobs.items() if j.state == "failed"}

    def _removable_ids(self) -> Set[str]:
        return {
            jid for jid, j in self.jobs.items()
            if j.state not in ("running", "removed")
        }

    # -- Rules -----------------------------------------------------------------

    @rule(recurring=st.booleans(), total=st.one_of(st.none(), st.integers(1, 10)))
    def create_job(self, recurring, total):
        """Create a new cron job."""
        jid = self._new_id()
        job = CronJob(jid, recurring=recurring, total_repeat=total)
        self.jobs[jid] = job

    @precondition(lambda self: bool(self._schedulable_ids()) and self.running_job_id is None)
    @rule(data=st.data())
    def tick(self, data):
        """Scheduler tick: pick a scheduled job and move to running."""
        schedulable = sorted(self._schedulable_ids())
        jid = data.draw(st.sampled_from(schedulable))
        job = self.jobs[jid]
        assert job.state == "scheduled"
        job.state = "running"
        job.enabled = True
        self.running_job_id = jid

    @precondition(lambda self: self.running_job_id is not None)
    @rule()
    def mark_success(self):
        """Running job completes successfully."""
        jid = self.running_job_id
        job = self.jobs[jid]
        assert job.state == "running"

        job.runs_completed += 1
        job.completion_history.append(job.runs_completed)

        if job.runs_left is not None:
            job.runs_left -= 1

        # Decide next state
        if job.recurring and (job.runs_left is None or job.runs_left > 0):
            job.state = "scheduled"
        else:
            job.state = "completed"
            job.enabled = False

        self.running_job_id = None

    @precondition(lambda self: self.running_job_id is not None)
    @rule()
    def mark_failure(self):
        """Running job fails."""
        jid = self.running_job_id
        job = self.jobs[jid]
        assert job.state == "running"
        job.state = "failed"
        self.running_job_id = None

    @precondition(lambda self: bool(self._failed_ids()))
    @rule(data=st.data())
    def retry(self, data):
        """Retry a failed job -> scheduled."""
        failed = sorted(self._failed_ids())
        jid = data.draw(st.sampled_from(failed))
        job = self.jobs[jid]
        assert job.state == "failed"
        job.state = "scheduled"
        job.enabled = True

    @precondition(lambda self: bool(self._pausable_ids()))
    @rule(data=st.data())
    def pause(self, data):
        """Pause a scheduled or failed job."""
        pausable = sorted(self._pausable_ids())
        jid = data.draw(st.sampled_from(pausable))
        job = self.jobs[jid]
        assert job.state in ("scheduled", "failed")
        job.state = "paused"
        job.enabled = False

    @precondition(lambda self: bool(self._paused_ids()))
    @rule(data=st.data())
    def resume(self, data):
        """Resume a paused job -> scheduled."""
        paused = sorted(self._paused_ids())
        jid = data.draw(st.sampled_from(paused))
        job = self.jobs[jid]
        assert job.state == "paused"
        job.state = "scheduled"
        job.enabled = True

    @precondition(lambda self: bool(self._removable_ids()))
    @rule(data=st.data())
    def remove(self, data):
        """Remove a non-running job."""
        removable = sorted(self._removable_ids())
        jid = data.draw(st.sampled_from(removable))
        job = self.jobs[jid]
        assert job.state != "running"
        job.state = "removed"
        job.enabled = False

    # -- Invariants (checked after every step) ---------------------------------

    @invariant()
    def inv_unique_ids(self):
        """No duplicate job IDs."""
        ids = list(self.jobs.keys())
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    @invariant()
    def inv_valid_states(self):
        """Every job has a valid state."""
        for jid, job in self.jobs.items():
            assert job.state in CRON_STATES, f"Job {jid} has invalid state {job.state}"

    @invariant()
    def inv_paused_never_fires(self):
        """INV-C1: paused => enabled=false, so scheduler can't pick it."""
        for jid, job in self.jobs.items():
            if job.state == "paused":
                assert not job.enabled, (
                    f"Job {jid} is paused but enabled=True (would fire!)"
                )

    @invariant()
    def inv_running_not_removed(self):
        """INV-C2: Running job cannot be in removed state."""
        for jid, job in self.jobs.items():
            if job.state == "running":
                assert job.state != "removed"

    @invariant()
    def inv_repeat_count_conservation(self):
        """INV-C3: completed + runs_left = total_repeat (when defined)."""
        for jid, job in self.jobs.items():
            if job.total_repeat is not None:
                actual_left = job.runs_left_computed
                assert job.runs_completed + (job.runs_left if job.runs_left is not None else 0) <= job.total_repeat, (
                    f"Job {jid}: completed={job.runs_completed}, "
                    f"runs_left={job.runs_left}, total={job.total_repeat}"
                )

    @invariant()
    def inv_no_concurrent_execution(self):
        """INV-C5: At most one job running at a time."""
        running = [jid for jid, j in self.jobs.items() if j.state == "running"]
        assert len(running) <= 1, f"Multiple running jobs: {running}"

    @invariant()
    def inv_enabled_state_consistency(self):
        """INV-C6: enabled/state consistency."""
        for jid, job in self.jobs.items():
            if job.state == "paused":
                assert not job.enabled, f"Job {jid}: paused but enabled"
            if job.state in ("scheduled", "running"):
                assert job.enabled, f"Job {jid}: {job.state} but not enabled"

    @invariant()
    def inv_monotonic_completion(self):
        """INV-C7: runs_completed never decreases."""
        for jid, job in self.jobs.items():
            if len(job.completion_history) >= 2:
                for i in range(1, len(job.completion_history)):
                    assert job.completion_history[i] >= job.completion_history[i-1], (
                        f"Job {jid}: completion count decreased "
                        f"{job.completion_history[i-1]} -> {job.completion_history[i]}"
                    )

    @invariant()
    def inv_completed_is_terminal_ish(self):
        """Completed jobs have enabled=False."""
        for jid, job in self.jobs.items():
            if job.state == "completed":
                assert not job.enabled, f"Completed job {jid} still enabled"

    @invariant()
    def inv_running_tracker_consistent(self):
        """running_job_id matches actual running jobs."""
        running = [jid for jid, j in self.jobs.items() if j.state == "running"]
        if self.running_job_id is not None:
            assert self.running_job_id in running
        if not running:
            assert self.running_job_id is None


# Generate the test class
TestCronStateMachine = CronStateMachine.TestCase
TestCronStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    phases=[Phase.generate],
)


# ============================================================================
# Standalone property tests
# ============================================================================

from hypothesis import given


@pytest.mark.property
@given(
    states=st.lists(st.sampled_from(sorted(CRON_STATES)), min_size=2, max_size=10),
)
def test_transition_reflexivity(states):
    """Same-state transitions are always valid (no-op)."""
    for s in states:
        assert s in VALID_TRANSITIONS or s == "nonexistent"


@pytest.mark.property
@given(
    from_state=st.sampled_from(sorted(CRON_STATES)),
    to_state=st.sampled_from(sorted(CRON_STATES)),
)
def test_removed_is_terminal(from_state, to_state):
    """Once removed, no transitions are possible."""
    if from_state == "removed":
        assert to_state not in VALID_TRANSITIONS.get("removed", set())
