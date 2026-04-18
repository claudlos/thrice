"""
Generic state machine framework for Hermes.

Provides a reusable StateMachine class with:
- Explicit states and transitions
- Guard conditions on transitions
- Post-transition invariant checking
- Transition logging and history
- Hierarchical state support (parent/child states)

Inspired by tla-precheck's DSL and sample-task-management's runtime checks.

Usage:
    from state_machine import StateMachine

    sm = StateMachine(
        name="example",
        states={"idle", "running", "done"},
        initial_state="idle",
        transitions={
            "start": [("idle", "running")],
            "finish": [("running", "done")],
        },
    )

    sm.apply("start")       # idle -> running
    sm.apply("finish")      # running -> done
    sm.apply("start")       # raises InvalidTransition
"""

import logging
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidTransition(Exception):
    """Raised when a transition is not valid from the current state."""

    def __init__(self, action: str, current_state: str, message: str = ""):
        self.action = action
        self.current_state = current_state
        super().__init__(
            message or f"Invalid transition: '{action}' from state '{current_state}'"
        )


class GuardFailed(InvalidTransition):
    """Raised when a transition guard condition returns False."""

    def __init__(self, action: str, current_state: str, guard_name: str = ""):
        self.guard_name = guard_name
        super().__init__(
            action,
            current_state,
            f"Guard failed for '{action}' from '{current_state}'"
            + (f": {guard_name}" if guard_name else ""),
        )


class InvariantViolation(Exception):
    """Raised when a post-transition invariant check fails."""

    def __init__(self, violations: List[str]):
        self.violations = violations
        super().__init__(
            f"State machine invariant violation(s): {'; '.join(violations)}"
        )


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------

@dataclass
class TransitionRecord:
    """Record of a state transition for logging/debugging."""
    timestamp: float
    action: str
    from_state: str
    to_state: str
    context: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f"Transition({self.action}: {self.from_state} -> {self.to_state} "
            f"at {self.timestamp:.3f})"
        )


# ---------------------------------------------------------------------------
# Transition definition
# ---------------------------------------------------------------------------

# A guard is a callable: (current_state, action, context) -> bool
GuardFn = Callable[[str, str, Dict[str, Any]], bool]

# An invariant is a callable: (state_machine) -> List[str] (violations)
InvariantFn = Callable[["StateMachine"], List[str]]

# Callback after transition: (record, state_machine) -> None
OnTransitionFn = Callable[[TransitionRecord, "StateMachine"], None]


@dataclass
class TransitionDef:
    """Definition of a single transition edge."""
    from_state: str
    to_state: str
    guard: Optional[GuardFn] = None
    guard_name: str = ""

    def __repr__(self) -> str:
        g = f" [{self.guard_name}]" if self.guard_name else ""
        return f"{self.from_state} -> {self.to_state}{g}"


# Wildcard constant for "any state" transitions (e.g., remove from ANY)
ANY_STATE = "__ANY__"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class StateMachine:
    """Generic state machine with guards, invariants, and transition logging.

    Supports:
    - Named actions that trigger transitions between states
    - Guard functions that must return True for a transition to proceed
    - Invariant functions checked after every transition
    - Transition history with timestamps
    - Hierarchical states via parent_states mapping
    - Wildcard (ANY) source state transitions
    """

    def __init__(
        self,
        name: str,
        states: Set[str],
        initial_state: str,
        transitions: Optional[Dict[str, List[Union[Tuple, TransitionDef]]]] = None,
        guards: Optional[Dict[str, GuardFn]] = None,
        invariants: Optional[List[InvariantFn]] = None,
        parent_states: Optional[Dict[str, str]] = None,
        on_transition: Optional[OnTransitionFn] = None,
        max_history: int = 1000,
        check_invariants_on_transition: bool = True,
    ):
        """Initialize the state machine.

        Args:
            name: Human-readable name for logging.
            states: Set of valid state names.
            initial_state: Starting state (must be in states).
            transitions: Dict mapping action names to list of (from, to) tuples
                         or TransitionDef objects.
            guards: Dict mapping "action:from->to" keys to guard functions.
                    Also accepts action-level guards as "action" keys.
            invariants: List of invariant check functions.
            parent_states: Dict mapping child_state -> parent_state for hierarchy.
            on_transition: Callback invoked after each successful transition.
            max_history: Maximum transition records to keep.
            check_invariants_on_transition: Whether to check invariants after each transition.
        """
        self.name = name
        self._states = frozenset(states)
        self._current_state = initial_state
        self._history: List[TransitionRecord] = []
        self._max_history = max_history
        self._on_transition = on_transition
        self._check_invariants = check_invariants_on_transition
        self._context: Dict[str, Any] = {}

        if initial_state not in self._states:
            raise ValueError(
                f"Initial state '{initial_state}' not in states: {states}"
            )

        # Parse transitions
        self._transitions: Dict[str, List[TransitionDef]] = {}
        if transitions:
            for action, edges in transitions.items():
                self._transitions[action] = []
                for edge in edges:
                    if isinstance(edge, TransitionDef):
                        self._transitions[action].append(edge)
                    elif isinstance(edge, (tuple, list)):
                        from_s, to_s = edge[0], edge[1]
                        guard = edge[2] if len(edge) > 2 else None
                        guard_name = edge[3] if len(edge) > 3 else ""
                        self._transitions[action].append(
                            TransitionDef(from_s, to_s, guard, guard_name)
                        )

        # Additional named guards
        self._named_guards: Dict[str, GuardFn] = guards or {}

        # Invariants
        self._invariants: List[InvariantFn] = invariants or []

        # Hierarchical state support
        self._parent_states: Dict[str, str] = parent_states or {}

    # ----- Properties -----

    @property
    def state(self) -> str:
        """Current state."""
        return self._current_state

    @property
    def states(self) -> FrozenSet[str]:
        """All valid states."""
        return self._states

    @property
    def history(self) -> List[TransitionRecord]:
        """Transition history."""
        return list(self._history)

    @property
    def context(self) -> Dict[str, Any]:
        """Mutable context dict for storing associated data."""
        return self._context

    # ----- State queries -----

    def is_in_state(self, state: str) -> bool:
        """Check if currently in the given state (or a child of it)."""
        if self._current_state == state:
            return True
        # Check hierarchy: is current state a descendant of `state`?
        current = self._current_state
        while current in self._parent_states:
            current = self._parent_states[current]
            if current == state:
                return True
        return False

    def get_parent_state(self, state: Optional[str] = None) -> Optional[str]:
        """Get the parent state of the given state (or current state)."""
        s = state or self._current_state
        return self._parent_states.get(s)

    def get_available_actions(self) -> List[str]:
        """Get all actions that have at least one valid transition from current state."""
        available = []
        for action, edges in self._transitions.items():
            for edge in edges:
                if edge.from_state == self._current_state or edge.from_state == ANY_STATE:
                    available.append(action)
                    break
        return available

    # ----- Transition validation -----

    def is_valid_transition(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if an action can be applied from the current state.

        Checks both structural validity (edge exists) and guard conditions.
        """
        try:
            self._resolve_transition(action, context or {})
            return True
        except (InvalidTransition, GuardFailed):
            return False

    def _resolve_transition(
        self,
        action: str,
        context: Dict[str, Any],
    ) -> TransitionDef:
        """Find the matching transition definition for an action.

        Returns the first matching TransitionDef.
        Raises InvalidTransition if no edge matches.
        Raises GuardFailed if edge exists but guard fails.
        """
        if action not in self._transitions:
            raise InvalidTransition(
                action,
                self._current_state,
                f"Unknown action '{action}' for state machine '{self.name}'",
            )

        edges = self._transitions[action]
        matching_edges = []

        for edge in edges:
            if edge.from_state == self._current_state or edge.from_state == ANY_STATE:
                # Check if to_state is valid (ANY_STATE edges may target specific states)
                if edge.to_state not in self._states:
                    continue
                matching_edges.append(edge)

        if not matching_edges:
            raise InvalidTransition(
                action,
                self._current_state,
                f"No transition for '{action}' from state '{self._current_state}' "
                f"in state machine '{self.name}'",
            )

        # Try edges in order, checking guards
        last_guard_failure = None
        for edge in matching_edges:
            # Check edge-level guard
            if edge.guard is not None:
                if not edge.guard(self._current_state, action, context):
                    last_guard_failure = edge.guard_name or "edge_guard"
                    continue

            # Check named guard (format: "action:from->to" or just "action")
            guard_key = f"{action}:{edge.from_state}->{edge.to_state}"
            if guard_key in self._named_guards:
                if not self._named_guards[guard_key](
                    self._current_state, action, context
                ):
                    last_guard_failure = guard_key
                    continue

            # Action-level guard
            if action in self._named_guards:
                if not self._named_guards[action](
                    self._current_state, action, context
                ):
                    last_guard_failure = action
                    continue

            return edge

        # All matching edges failed their guards
        raise GuardFailed(
            action,
            self._current_state,
            last_guard_failure or "unknown",
        )

    # ----- State transitions -----

    def apply(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TransitionRecord:
        """Apply an action, transitioning to a new state.

        Args:
            action: The action/event name.
            context: Optional context dict passed to guards and callbacks.

        Returns:
            TransitionRecord of the completed transition.

        Raises:
            InvalidTransition: If no valid transition exists.
            GuardFailed: If guard conditions fail.
            InvariantViolation: If post-transition invariants fail.
        """
        ctx = context or {}
        edge = self._resolve_transition(action, ctx)

        old_state = self._current_state
        new_state = edge.to_state

        # Apply transition
        self._current_state = new_state

        # Record
        record = TransitionRecord(
            timestamp=time.time(),
            action=action,
            from_state=old_state,
            to_state=new_state,
            context=ctx if ctx else None,
        )
        self._history.append(record)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Log
        logger.debug(
            "[%s] %s: %s -> %s",
            self.name,
            action,
            old_state,
            new_state,
        )

        # Callback
        if self._on_transition:
            self._on_transition(record, self)

        # Invariant checks
        if self._check_invariants:
            violations = self.check_invariants()
            if violations:
                # Roll back on violation (best effort)
                self._current_state = old_state
                logger.error(
                    "[%s] Invariant violation after %s: %s -> %s: %s",
                    self.name,
                    action,
                    old_state,
                    new_state,
                    violations,
                )
                raise InvariantViolation(violations)

        return record

    def force_state(self, state: str, reason: str = "") -> None:
        """Force the machine into a specific state (for recovery/testing).

        This bypasses guards and transition validation. Use sparingly.
        """
        if state not in self._states:
            raise ValueError(f"State '{state}' not in valid states: {self._states}")

        old = self._current_state
        self._current_state = state

        record = TransitionRecord(
            timestamp=time.time(),
            action=f"__force__({reason})" if reason else "__force__",
            from_state=old,
            to_state=state,
        )
        self._history.append(record)

        logger.warning(
            "[%s] Force state: %s -> %s (reason: %s)",
            self.name,
            old,
            state,
            reason or "none",
        )

    # ----- Invariant checking -----

    def check_invariants(self) -> List[str]:
        """Run all registered invariant functions.

        Returns a list of violation description strings (empty = all OK).
        """
        violations = []
        for inv_fn in self._invariants:
            try:
                result = inv_fn(self)
                if result:
                    violations.extend(result)
            except Exception as e:
                violations.append(f"INVARIANT_ERROR: {inv_fn.__name__}: {e}")
        return violations

    def add_invariant(self, fn: InvariantFn) -> None:
        """Register an additional invariant check function."""
        self._invariants.append(fn)

    def remove_invariant(self, fn: InvariantFn) -> None:
        """Remove a registered invariant check function."""
        self._invariants = [f for f in self._invariants if f is not fn]

    # ----- Transition management -----

    def add_transition(
        self,
        action: str,
        from_state: str,
        to_state: str,
        guard: Optional[GuardFn] = None,
        guard_name: str = "",
    ) -> None:
        """Add a transition edge."""
        if action not in self._transitions:
            self._transitions[action] = []
        self._transitions[action].append(
            TransitionDef(from_state, to_state, guard, guard_name)
        )

    def add_guard(self, key: str, guard: GuardFn) -> None:
        """Add a named guard function.

        Key format: "action" or "action:from->to"
        """
        self._named_guards[key] = guard

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize machine state for debugging/persistence."""
        return {
            "name": self.name,
            "current_state": self._current_state,
            "states": sorted(self._states),
            "history_length": len(self._history),
            "available_actions": self.get_available_actions(),
            "context": self._context,
        }

    def __repr__(self) -> str:
        actions = ", ".join(self.get_available_actions()) or "none"
        return (
            f"StateMachine({self.name}, "
            f"state={self._current_state}, "
            f"available=[{actions}])"
        )
