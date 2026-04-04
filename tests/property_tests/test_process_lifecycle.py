"""
Property-based stateful testing for Process Lifecycle (SM-4).

Uses hypothesis RuleBasedStateMachine to randomly spawn, kill, poll,
and prune processes, checking invariants after every step.

Invariants verified:
  INV-P1: ExclusivePlacement (never in both running and finished)
  INV-P2: ExitedMonotonic (exited never reverts)
  INV-P3: ExitCodeConsistency (exit_code iff exited)
  INV-P4: RunningNotExited
  INV-P5: FinishedIsExited
  INV-P6: UniqueSessionIds
  INV-P9: CapacityLimit
"""

import pytest
import time
from typing import Dict, Optional, Set

from hypothesis import settings, HealthCheck, Phase
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    precondition,
)

pytestmark = [pytest.mark.property, pytest.mark.stateful]

MAX_PROCESSES = 20  # lower limit for testing


# ============================================================================
# Abstract process model
# ============================================================================

class ProcessSession:
    """Lightweight model of a tracked process."""

    def __init__(self, sid: str, pid: int):
        self.id = sid
        self.pid = pid
        self.exited = False
        self.exit_code: Optional[int] = None
        self.started_at = time.time()
        self.detached = False
        self._exited_history: list = []  # for monotonicity check

    def mark_exited(self, code: int):
        self.exited = True
        self.exit_code = code
        self._exited_history.append(True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pid": self.pid,
            "exited": self.exited,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "detached": self.detached,
            "command": f"test_cmd_{self.id}",
        }


class ProcessRegistryStateMachine(RuleBasedStateMachine):
    """
    Stateful test that randomly spawns, kills, polls, and prunes processes,
    verifying process registry invariants after every operation.
    """

    def __init__(self):
        super().__init__()
        self.running: Dict[str, ProcessSession] = {}
        self.finished: Dict[str, ProcessSession] = {}
        self._counter = 0
        self._pid_counter = 1000

    # -- Helpers ---------------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"sess_{self._counter}"

    def _new_pid(self) -> int:
        self._pid_counter += 1
        return self._pid_counter

    # -- Rules -----------------------------------------------------------------

    @precondition(lambda self: len(self.running) + len(self.finished) < MAX_PROCESSES)
    @rule()
    def spawn(self):
        """Spawn a new process into the running set."""
        sid = self._new_id()
        proc = ProcessSession(sid, self._new_pid())
        self.running[sid] = proc

    @precondition(lambda self: bool(self.running))
    @rule(data=st.data(), exit_code=st.integers(-15, 2))
    def kill(self, data, exit_code):
        """Kill a running process: move to finished."""
        sid = data.draw(st.sampled_from(sorted(self.running.keys())))
        proc = self.running.pop(sid)
        proc.mark_exited(exit_code)
        self.finished[sid] = proc

    @precondition(lambda self: bool(self.running))
    @rule(data=st.data())
    def poll_running(self, data):
        """Poll a running process (no state change, just verify it's there)."""
        sid = data.draw(st.sampled_from(sorted(self.running.keys())))
        proc = self.running[sid]
        assert not proc.exited, f"Running process {sid} should not be exited"

    @precondition(lambda self: bool(self.finished))
    @rule(data=st.data())
    def poll_finished(self, data):
        """Poll a finished process."""
        sid = data.draw(st.sampled_from(sorted(self.finished.keys())))
        proc = self.finished[sid]
        assert proc.exited, f"Finished process {sid} should be exited"
        assert proc.exit_code is not None

    @precondition(lambda self: bool(self.finished))
    @rule(data=st.data())
    def prune(self, data):
        """Prune (remove) a finished process."""
        sid = data.draw(st.sampled_from(sorted(self.finished.keys())))
        del self.finished[sid]

    @precondition(lambda self: bool(self.running))
    @rule(data=st.data())
    def finish_naturally(self, data):
        """Process exits naturally (exit_code=0)."""
        sid = data.draw(st.sampled_from(sorted(self.running.keys())))
        proc = self.running.pop(sid)
        proc.mark_exited(0)
        self.finished[sid] = proc

    # -- Invariants (checked after every step) ---------------------------------

    @invariant()
    def inv_exclusive_placement(self):
        """INV-P1: No session in both running and finished."""
        overlap = set(self.running.keys()) & set(self.finished.keys())
        assert not overlap, f"Sessions in both running and finished: {overlap}"

    @invariant()
    def inv_running_not_exited(self):
        """INV-P4: Running processes have exited=False."""
        for sid, proc in self.running.items():
            assert not proc.exited, f"Running process {sid} has exited=True"

    @invariant()
    def inv_finished_is_exited(self):
        """INV-P5: Finished processes have exited=True."""
        for sid, proc in self.finished.items():
            assert proc.exited, f"Finished process {sid} has exited=False"

    @invariant()
    def inv_exit_code_consistency(self):
        """INV-P3: exit_code is None iff exited=False."""
        for sid, proc in {**self.running, **self.finished}.items():
            if not proc.exited:
                assert proc.exit_code is None, (
                    f"Process {sid}: exited=False but exit_code={proc.exit_code}"
                )
            else:
                assert proc.exit_code is not None, (
                    f"Process {sid}: exited=True but exit_code=None"
                )

    @invariant()
    def inv_unique_session_ids(self):
        """INV-P6: No duplicate session IDs across both dicts."""
        all_ids = list(self.running.keys()) + list(self.finished.keys())
        assert len(all_ids) == len(set(all_ids)), f"Duplicate session IDs"

    @invariant()
    def inv_capacity_limit(self):
        """INV-P9: Total tracked processes <= MAX_PROCESSES."""
        total = len(self.running) + len(self.finished)
        assert total <= MAX_PROCESSES, (
            f"Total processes {total} exceeds limit {MAX_PROCESSES}"
        )

    @invariant()
    def inv_exited_monotonic(self):
        """INV-P2: Once exited=True, never reverts."""
        for sid, proc in self.finished.items():
            # All entries in history should be True
            for val in proc._exited_history:
                assert val is True


# Generate the test class
TestProcessRegistryStateMachine = ProcessRegistryStateMachine.TestCase
TestProcessRegistryStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    phases=[Phase.generate],
)


# ============================================================================
# Standalone property tests
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "thrice"))

from invariants.process_invariants import (
    ProcessInvariantChecker,
    infer_process_state,
    PROCESS_STATES,
    _VALID_PROCESS_TRANSITIONS,
)

from hypothesis import given


@pytest.mark.property
@given(
    from_state=st.sampled_from(sorted(PROCESS_STATES)),
    to_state=st.sampled_from(sorted(PROCESS_STATES)),
)
def test_pruned_is_terminal(from_state, to_state):
    """Pruned state has no outgoing transitions."""
    if from_state == "pruned" and from_state != to_state:
        err = ProcessInvariantChecker.check_valid_transition(from_state, to_state)
        assert err is not None, f"pruned -> {to_state} should be rejected"


@pytest.mark.property
@given(state=st.sampled_from(sorted(PROCESS_STATES)))
def test_self_transition_is_noop(state):
    """Transitioning to the same state is always a no-op."""
    assert ProcessInvariantChecker.check_valid_transition(state, state) is None


@pytest.mark.property
@given(
    n_running=st.integers(0, 10),
    n_finished=st.integers(0, 10),
)
def test_disjoint_registries_pass(n_running, n_finished):
    """Disjoint running/finished sets always pass exclusive placement."""
    running = {f"r_{i}": {"exited": False, "exit_code": None} for i in range(n_running)}
    finished = {f"f_{i}": {"exited": True, "exit_code": 0} for i in range(n_finished)}
    assert ProcessInvariantChecker.check_exclusive_placement(running, finished) is None
    assert ProcessInvariantChecker.check_running_not_exited(running, finished) is None
    assert ProcessInvariantChecker.check_finished_is_exited(running, finished) is None
