"""
Tests for ``modules/state_machine.py`` - the generic StateMachine framework
that every Thrice SM (cron, agent loop, etc.) builds on.

These tests focus on correctness of the *framework* itself, independent
of any concrete SM.  They pair with the refinement tests in
``tests/refinement_tests/test_thrice_sm_refinement.py`` which exercise
the concrete machines against their TLA+ specifications.
"""
from __future__ import annotations

import os
import sys
from typing import List

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from state_machine import (  # noqa: E402
    ANY_STATE,
    GuardFailed,
    InvalidTransition,
    InvariantViolation,
    StateMachine,
    TransitionDef,
    TransitionRecord,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sm() -> StateMachine:
    """A simple 3-state SM with guarded transitions."""
    def guard_ready(current, action, ctx):
        return ctx.get("ready", False)

    transitions = {
        "start": [TransitionDef("idle", "running", guard_ready, "ready")],
        "finish": [TransitionDef("running", "done")],
        "reset": [TransitionDef(ANY_STATE, "idle")],
    }
    return StateMachine(
        name="test",
        states={"idle", "running", "done"},
        initial_state="idle",
        transitions=transitions,
    )


# -----------------------------------------------------------------------------
# Construction & validation
# -----------------------------------------------------------------------------

class TestConstruction:

    def test_initial_state_must_be_in_states(self):
        with pytest.raises(ValueError):
            StateMachine(
                name="bad",
                states={"a", "b"},
                initial_state="c",   # not in the set
            )

    def test_empty_transitions_is_legal(self):
        m = StateMachine(name="stub", states={"a"}, initial_state="a")
        assert m.state == "a"
        assert m.get_available_actions() == []

    def test_tuple_shorthand_is_accepted(self):
        """Transitions can be given as (from, to) tuples instead of TransitionDef."""
        m = StateMachine(
            name="tuple",
            states={"a", "b"},
            initial_state="a",
            transitions={"go": [("a", "b")]},
        )
        m.apply("go")
        assert m.state == "b"


# -----------------------------------------------------------------------------
# Happy-path transitions
# -----------------------------------------------------------------------------

class TestTransitions:

    def test_guard_allows_transition(self, sm):
        rec = sm.apply("start", context={"ready": True})
        assert sm.state == "running"
        assert isinstance(rec, TransitionRecord)
        assert rec.from_state == "idle"
        assert rec.to_state == "running"

    def test_guard_blocks_transition(self, sm):
        with pytest.raises(GuardFailed):
            sm.apply("start", context={"ready": False})
        assert sm.state == "idle"   # unchanged on guard failure

    def test_unknown_action_raises(self, sm):
        with pytest.raises(InvalidTransition):
            sm.apply("teleport")

    def test_invalid_transition_from_current_state(self, sm):
        # "finish" is only valid from "running"
        with pytest.raises(InvalidTransition):
            sm.apply("finish")

    def test_wildcard_source_allows_any_state(self, sm):
        sm.apply("start", context={"ready": True})
        sm.apply("finish")
        assert sm.state == "done"
        sm.apply("reset")
        assert sm.state == "idle"


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------

class TestHistory:

    def test_history_records_all_transitions(self, sm):
        sm.apply("start", context={"ready": True})
        sm.apply("finish")
        assert [h.to_state for h in sm.history] == ["running", "done"]

    def test_history_respects_max(self):
        m = StateMachine(
            name="hist",
            states={"a", "b"},
            initial_state="a",
            transitions={"flip": [("a", "b"), ("b", "a")]},
            max_history=3,
        )
        for _ in range(5):
            m.apply("flip")
        assert len(m.history) <= 3


# -----------------------------------------------------------------------------
# Invariants
# -----------------------------------------------------------------------------

class TestInvariants:

    def _make_inv(self, msg: str = ""):
        """Factory: invariant that fires iff ``ctx.fail == True``."""
        def inv(sm: StateMachine) -> List[str]:
            if sm.context.get("fail"):
                return [msg or "fail flag set"]
            return []
        return inv

    def test_invariant_violation_raises(self):
        m = StateMachine(
            name="inv",
            states={"a", "b"},
            initial_state="a",
            transitions={"go": [("a", "b")]},
            invariants=[self._make_inv("no fail in b")],
        )
        m.context["fail"] = True
        with pytest.raises(InvariantViolation):
            m.apply("go")
        # Rolled back after the violation.
        assert m.state == "a"

    def test_invariant_pass_commits_transition(self):
        m = StateMachine(
            name="inv",
            states={"a", "b"},
            initial_state="a",
            transitions={"go": [("a", "b")]},
            invariants=[self._make_inv()],
        )
        m.apply("go")
        assert m.state == "b"

    def test_check_invariants_returns_violations_list(self):
        m = StateMachine(
            name="inv",
            states={"a"},
            initial_state="a",
            invariants=[
                lambda _sm: ["inv-1 failed"],
                lambda _sm: [],
                lambda _sm: ["inv-3 failed"],
            ],
        )
        violations = m.check_invariants()
        assert violations == ["inv-1 failed", "inv-3 failed"]


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

class TestCallbacks:

    def test_on_transition_is_called(self):
        events: list = []

        def cb(record, sm):
            events.append((record.from_state, record.to_state, sm.state))

        m = StateMachine(
            name="cb",
            states={"a", "b"},
            initial_state="a",
            transitions={"go": [("a", "b")]},
            on_transition=cb,
        )
        m.apply("go")
        assert events == [("a", "b", "b")]


# -----------------------------------------------------------------------------
# force_state
# -----------------------------------------------------------------------------

class TestForceState:

    def test_force_state_bypasses_transitions(self, sm):
        sm.force_state("done", reason="test fixture setup")
        assert sm.state == "done"
        # History should still record it (action string is implementation-
        # defined but must contain "force").
        assert "force" in sm.history[-1].action.lower()

    def test_force_state_rejects_unknown(self, sm):
        with pytest.raises(ValueError):
            sm.force_state("nope")
