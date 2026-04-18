"""
Refinement tests: Thrice concrete state machines vs TLA+ abstract specs.

These tests verify, *without* requiring Hermes to be installed, that the
state machines shipped in `modules/` refine the abstract specifications
under `specs/tla/`.  The abstract transition tables encoded below are
the Python mirror of `CronJob.tla` and `AgentLoop.tla` — if you change
one side you must change the other, and these tests will fail until
they match.

Refinement = simulation relation:
  1. Every concrete state maps to an abstract state.
  2. Every concrete transition is allowed in the abstract transition
     table (possibly under a guard).
  3. Abstract invariants hold on every reached concrete state.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, FrozenSet

import pytest

# Allow running with `PYTHONPATH=modules:.` or pip-installed layout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from agent_loop_state_machine import (  # noqa: E402
    AgentLoopState,
    AgentLoopStateMachine,
)
from cron_state_machine import CRON_STATES, CronJobStateMachine  # noqa: E402

pytestmark = pytest.mark.refinement


# =============================================================================
# Abstract SM-1 transition table (must match CronJob.tla)
# =============================================================================

ABSTRACT_CRON_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "nonexistent": frozenset({"scheduled", "removed"}),
    "scheduled":   frozenset({"running", "paused", "removed"}),
    "running":     frozenset({"scheduled", "completed", "failed"}),
    "paused":      frozenset({"scheduled", "removed"}),
    "completed":   frozenset({"removed"}),
    "failed":      frozenset({"scheduled", "paused", "removed"}),
    "removed":     frozenset(),                    # absorbing
}


class TestCronRefinement:
    """SM-1: CronJobStateMachine refines CronJob.tla."""

    def test_states_match_abstract(self):
        assert set(CRON_STATES) == set(ABSTRACT_CRON_TRANSITIONS.keys())

    def test_create_transition(self):
        sm = CronJobStateMachine(job_id="t")
        assert sm.state == "nonexistent"
        sm.create(next_run_at=datetime.now(timezone.utc) + timedelta(seconds=1))
        assert sm.state == "scheduled"
        assert sm.state in ABSTRACT_CRON_TRANSITIONS["nonexistent"]

    def test_tick_transition(self):
        sm = CronJobStateMachine(job_id="t")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        sm.create(next_run_at=past)
        sm.tick(now=datetime.now(timezone.utc))
        assert sm.state == "running"
        assert sm.state in ABSTRACT_CRON_TRANSITIONS["scheduled"]

    def test_mark_success_one_shot(self):
        sm = CronJobStateMachine(job_id="t")
        sm.create(next_run_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        sm.tick(now=datetime.now(timezone.utc))
        sm.mark_success()
        assert sm.state == "completed"
        assert sm.state in ABSTRACT_CRON_TRANSITIONS["running"]

    def test_mark_success_recurring_runs_left(self):
        sm = CronJobStateMachine(job_id="t")
        sm.create(
            next_run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            recurring=True,
            repeat_total=3,
        )
        sm.tick(now=datetime.now(timezone.utc))
        sm.mark_success(next_run_at=datetime.now(timezone.utc) + timedelta(seconds=1))
        assert sm.state == "scheduled"

    def test_retry_bounded(self):
        """TLA+ invariant RetryBounded: retryCount <= maxRetries + 1.

        The guard on ``Retry`` is ``retry_count < max_retries``.  With
        max_retries=2 we can retry once (count 1), but the second retry
        attempt (count 2) must be refused by the guard.
        """
        from state_machine import GuardFailed, InvalidTransition

        sm = CronJobStateMachine(
            job_id="t",
            initial_state="nonexistent",
        )
        sm.create(next_run_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                  max_retries=2)
        sm.tick(now=datetime.now(timezone.utc))
        sm.mark_failure()                                   # retry_count = 1
        sm.retry(next_run_at=datetime.now(timezone.utc))    # guard 1<2: OK
        assert sm.state == "scheduled"
        sm.tick(now=datetime.now(timezone.utc))
        sm.mark_failure()                                   # retry_count = 2
        with pytest.raises((GuardFailed, InvalidTransition)):
            sm.retry(next_run_at=datetime.now(timezone.utc))  # guard 2<2: refused

    def test_remove_blocked_from_running(self):
        sm = CronJobStateMachine(job_id="t")
        sm.create(next_run_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        sm.tick(now=datetime.now(timezone.utc))
        from state_machine import GuardFailed, InvalidTransition
        with pytest.raises((GuardFailed, InvalidTransition)):
            sm.remove()

    def test_all_invariants_hold_on_random_runs(self):
        """Drive an unbounded recurring job through several cycles and check
        the TLA+ invariants at every step.

        Unbounded = ``repeat_total=None``; ``_invariant_repeat_consistency``
        skips when total is None, matching the TLA+ precondition
        ``repeatTotal > 0`` in ``RepeatConsistency``.
        """
        sm = CronJobStateMachine(job_id="t")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)

        ops = [
            ("create", dict(recurring=True, repeat_total=None, next_run_at=past)),
            ("tick", dict(now=datetime.now(timezone.utc))),
            ("mark_success", dict(next_run_at=past)),
            ("tick", dict(now=datetime.now(timezone.utc))),
            ("mark_success", dict(next_run_at=past)),
            ("pause", {}),          # scheduled -> paused
            ("resume", dict(next_run_at=past)),   # paused -> scheduled
            ("remove", {}),         # scheduled -> removed
        ]
        for name, kwargs in ops:
            getattr(sm, name)(**kwargs)
            violations = sm.check_invariants()
            assert not violations, (name, violations)
        assert sm.state == "removed"


# =============================================================================
# Abstract SM-2 transition table (must match AgentLoop.tla)
# =============================================================================

S = AgentLoopState

ABSTRACT_AGENT_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    S.AWAITING_INPUT.value:        frozenset({S.PREPARING_API_CALL.value,
                                               S.INTERRUPTED.value}),
    S.PREPARING_API_CALL.value:    frozenset({S.CALLING_API.value,
                                               S.BUDGET_EXHAUSTED.value,
                                               S.INTERRUPTED.value}),
    S.CALLING_API.value:           frozenset({S.PROCESSING_RESPONSE.value,
                                               S.ERROR_RECOVERY.value,
                                               S.INTERRUPTED.value}),
    S.PROCESSING_RESPONSE.value:   frozenset({S.EXECUTING_TOOLS.value,
                                               S.HANDLING_CONTINUATION.value,
                                               S.INTERRUPTED.value}),
    S.EXECUTING_TOOLS.value:       frozenset({S.HANDLING_CONTINUATION.value,
                                               S.INTERRUPTED.value}),
    S.HANDLING_CONTINUATION.value: frozenset({S.PREPARING_API_CALL.value,
                                               S.COMPRESSING_CONTEXT.value,
                                               S.RETURNING_RESPONSE.value,
                                               S.BUDGET_EXHAUSTED.value,
                                               S.INTERRUPTED.value}),
    S.COMPRESSING_CONTEXT.value:   frozenset({S.PREPARING_API_CALL.value,
                                               S.INTERRUPTED.value}),
    S.ERROR_RECOVERY.value:        frozenset({S.CALLING_API.value,
                                               S.RETURNING_RESPONSE.value,
                                               S.INTERRUPTED.value}),
    S.RETURNING_RESPONSE.value:    frozenset(),       # terminal
    S.INTERRUPTED.value:           frozenset(),       # terminal
    S.BUDGET_EXHAUSTED.value:      frozenset(),       # terminal
}


class TestAgentLoopRefinement:
    """SM-2: AgentLoopStateMachine refines AgentLoop.tla."""

    def test_all_enum_values_have_abstract_edges(self):
        assert {s.value for s in AgentLoopState} == set(ABSTRACT_AGENT_TRANSITIONS.keys())

    def test_happy_path_text_only(self):
        """A happy text-only response walks through the expected abstract states."""
        sm = AgentLoopStateMachine(iteration_budget=3, max_retries=2)
        before = sm.state.value

        sm.receive_message(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "hi"}],
            system_prompt="s",
        )
        assert sm.state.value in ABSTRACT_AGENT_TRANSITIONS[before]
        before = sm.state.value

        sm.build_request({"model": "x", "messages": []})
        assert sm.state.value in ABSTRACT_AGENT_TRANSITIONS[before]
        before = sm.state.value

        sm.receive_response({"text": "hello"}, finish_reason="stop")
        assert sm.state.value in ABSTRACT_AGENT_TRANSITIONS[before]
        before = sm.state.value

        sm.process_text_response()
        assert sm.state.value in ABSTRACT_AGENT_TRANSITIONS[before]
        before = sm.state.value

        sm.return_result()
        assert sm.state == AgentLoopState.RETURNING_RESPONSE
        assert not ABSTRACT_AGENT_TRANSITIONS[sm.state.value], "terminal"

    def test_error_then_retry_path(self):
        sm = AgentLoopStateMachine(iteration_budget=5, max_retries=2)
        sm.receive_message(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "hi"}],
            system_prompt="s",
        )
        sm.build_request({"model": "x", "messages": []})
        assert sm.state == AgentLoopState.CALLING_API

        sm.recover_error(RuntimeError("boom"))
        assert sm.state == AgentLoopState.ERROR_RECOVERY
        sm.retry_api()
        assert sm.state == AgentLoopState.CALLING_API
        assert sm.loop_context.retry_count == 1

    def test_budget_exhausted_is_terminal(self):
        """LV-A3: exhausting the iteration budget lands in a terminal state."""
        sm = AgentLoopStateMachine(iteration_budget=1, max_retries=2)
        sm.receive_message(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "hi"}],
            system_prompt="s",
        )
        sm.build_request({"model": "x", "messages": []})
        sm.receive_response({"text": ""}, finish_reason="stop")
        sm.process_text_response()
        # Simulate the natural end-of-iteration: iterations_used == budget
        # (INV-A4 allows budget + 1, so budget is fine).
        assert sm.loop_context.iterations_used == 1
        assert not sm.loop_context.has_budget
        action = sm.decide_after_continuation()
        assert action == "exhaust_budget"
        sm.exhaust_budget()
        assert sm.state == AgentLoopState.BUDGET_EXHAUSTED
        # Terminal: no outgoing edges in the abstract transition table.
        assert not ABSTRACT_AGENT_TRANSITIONS[sm.state.value]

    def test_interrupt_reachable_from_non_terminal(self):
        """LV-A2 spot-check: interrupt from the middle of the loop lands in INTERRUPTED."""
        sm = AgentLoopStateMachine(iteration_budget=5, max_retries=2)
        sm.receive_message(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "hi"}],
            system_prompt="s",
        )
        sm.build_request({"model": "x", "messages": []})
        sm.interrupt()
        assert sm.state == AgentLoopState.INTERRUPTED

    def test_invariants_hold_on_scripted_run(self):
        sm = AgentLoopStateMachine(iteration_budget=4, max_retries=2)
        for step in [
            lambda: sm.receive_message(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "hi"}],
            system_prompt="s",
        ),
            lambda: sm.build_request({"model": "x"}),
            lambda: sm.receive_response({}, finish_reason="stop"),
            lambda: sm.process_text_response(),
            lambda: sm.return_result(),
        ]:
            step()
            violations = sm.check_invariants()
            assert not violations, (sm.state, violations)
        assert sm.state == AgentLoopState.RETURNING_RESPONSE
