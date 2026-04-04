"""
Shadow Mode for the Agent Loop State Machine (Phase 1 of migration).

Runs AgentLoopStateMachine as a passive observer alongside the existing
boolean-flag-driven loop in run_agent.py. Zero risk: all operations are
wrapped in try/except so the shadow never crashes the real loop.

The shadow tracks flag changes, mirrors them as state machine transitions,
and logs divergences between the SM state and the actual boolean flags.
This surfaces bugs in either the old code or the new SM before any
control flow is changed.

Usage (inside run_agent.py):
    from agent_loop_shadow import AgentLoopShadow

    shadow = AgentLoopShadow(iteration_budget=self.max_iterations)
    shadow.mirror_flag_change("messages_received", True, messages=messages)
    # ... throughout the loop ...
    divergence = shadow.check_divergence(current_flags_dict)
    if divergence:
        logger.warning("SM DIVERGENCE: %s", divergence)

See also: agent_loop_migration.md (Phase 1)
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_loop_state_machine import (
    Action,
    AgentLoopContext,
    AgentLoopState,
    AgentLoopStateMachine,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flag-to-state mapping
# ---------------------------------------------------------------------------

# Maps boolean flag names (as used in run_agent.py) to the AgentLoopState
# that should be active when the flag is True.
_FLAG_TO_STATE = {
    "restart_with_compressed_messages": AgentLoopState.COMPRESSING_CONTEXT,
    "restart_with_length_continuation": AgentLoopState.HANDLING_CONTINUATION,
    "budget_exceeded": AgentLoopState.BUDGET_EXHAUSTED,
    "interrupted": AgentLoopState.INTERRUPTED,
}

# Reverse mapping: state -> flag name(s) that should be True
_STATE_TO_FLAGS: Dict[AgentLoopState, Set[str]] = {}
for _flag, _state in _FLAG_TO_STATE.items():
    _STATE_TO_FLAGS.setdefault(_state, set()).add(_flag)


def _derive_expected_state(flags: Dict[str, Any]) -> AgentLoopState:
    """Derive the expected SM state from the current boolean flags.

    This is the core mapping that lets us compare the old flag-based
    control flow to the new state machine.

    Priority order matches the if/elif chain in run_agent.py:
        1. interrupted        -> INTERRUPTED
        2. budget_exceeded    -> BUDGET_EXHAUSTED
        3. restart_with_compressed_messages -> COMPRESSING_CONTEXT
        4. restart_with_length_continuation -> HANDLING_CONTINUATION
        5. has_tool_calls     -> EXECUTING_TOOLS
        6. response received  -> PROCESSING_RESPONSE
        7. in retry loop      -> ERROR_RECOVERY
        8. api call in flight -> CALLING_API
        9. building params    -> PREPARING_API_CALL
        10. default           -> AWAITING_INPUT (or RETURNING_RESPONSE if done)
    """
    if flags.get("interrupted"):
        return AgentLoopState.INTERRUPTED

    if flags.get("budget_exceeded"):
        return AgentLoopState.BUDGET_EXHAUSTED

    if flags.get("restart_with_compressed_messages"):
        return AgentLoopState.COMPRESSING_CONTEXT

    if flags.get("restart_with_length_continuation"):
        return AgentLoopState.HANDLING_CONTINUATION

    if flags.get("executing_tools"):
        return AgentLoopState.EXECUTING_TOOLS

    if flags.get("has_response") and not flags.get("response_processed"):
        return AgentLoopState.PROCESSING_RESPONSE

    if flags.get("in_retry"):
        return AgentLoopState.ERROR_RECOVERY

    if flags.get("api_call_in_flight"):
        return AgentLoopState.CALLING_API

    if flags.get("preparing_call"):
        return AgentLoopState.PREPARING_API_CALL

    if flags.get("loop_exited"):
        return AgentLoopState.RETURNING_RESPONSE

    return AgentLoopState.AWAITING_INPUT


# ---------------------------------------------------------------------------
# Divergence record
# ---------------------------------------------------------------------------

@dataclass
class DivergenceRecord:
    """Captures a single divergence between SM state and boolean flags."""
    timestamp: float
    sm_state: str
    expected_state: str
    flags: Dict[str, Any]
    context: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.timestamp:.3f}] SM={self.sm_state} "
            f"expected={self.expected_state} "
            f"flags={self.flags}"
            + (f" ctx={self.context}" if self.context else "")
        )


# ---------------------------------------------------------------------------
# AgentLoopShadow
# ---------------------------------------------------------------------------

class AgentLoopShadow:
    """Shadow-mode wrapper for AgentLoopStateMachine.

    Runs the state machine as a passive observer. All public methods
    are wrapped in try/except so the shadow never crashes the real loop.

    Attributes:
        sm: The underlying AgentLoopStateMachine.
        divergences: List of all divergences detected.
        transition_log: Structured log of all transitions.
        enabled: Kill switch to disable shadow mode at runtime.
    """

    def __init__(
        self,
        iteration_budget: int = 25,
        max_retries: int = 3,
        max_tokens: int = 200_000,
        compression_threshold: int = 150_000,
        check_invariants: bool = True,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.divergences: List[DivergenceRecord] = []
        self.transition_log: List[Dict[str, Any]] = []
        self._invariant_violations: List[Dict[str, Any]] = []
        self._error_count: int = 0
        self._transition_count: int = 0

        try:
            self.sm = AgentLoopStateMachine(
                iteration_budget=iteration_budget,
                max_retries=max_retries,
                max_tokens=max_tokens,
                compression_threshold=compression_threshold,
                check_invariants=check_invariants,
            )
        except Exception as e:
            logger.error("Shadow SM failed to initialize: %s", e)
            self.sm = None
            self.enabled = False

    # ---- Safe wrapper ----

    def _safe(self, fn_name: str, fn, *args, **kwargs) -> Any:
        """Execute fn in a try/except. Shadow never crashes the real loop."""
        if not self.enabled or self.sm is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self._error_count += 1
            logger.debug(
                "Shadow SM error in %s: %s (total errors: %d)",
                fn_name, e, self._error_count,
            )
            # Auto-disable if too many errors (something is fundamentally wrong)
            if self._error_count >= 50:
                logger.warning(
                    "Shadow SM disabled after %d errors", self._error_count
                )
                self.enabled = False
            return None

    # ---- Transition mirroring ----

    def mirror_flag_change(
        self,
        flag_name: str,
        value: Any,
        **extra_context,
    ) -> Optional[str]:
        """Mirror a boolean flag change to the state machine.

        Maps flag changes from run_agent.py to SM transitions.
        Returns the new SM state (as string) or None if no transition occurred.

        Supported flag_name values:
            "messages_received"  -> Action.RECEIVE_MESSAGE
            "preparing_call"     -> (handled by mirror_phase)
            "api_call_started"   -> Action.BUILD_REQUEST + SEND_API_CALL
            "response_received"  -> Action.RECEIVE_RESPONSE
            "api_error"          -> Action.RECOVER_ERROR
            "tool_dispatch"      -> Action.DISPATCH_TOOLS
            "tools_complete"     -> Action.TOOL_COMPLETE
            "text_response"      -> process_text_response
            "restart_with_compressed_messages" -> Action.COMPRESS
            "restart_with_length_continuation" -> Action.CONTINUE_GENERATION
            "budget_exceeded"    -> Action.EXHAUST_BUDGET
            "interrupted"        -> Action.INTERRUPT
            "return_result"      -> Action.RETURN_RESULT
            "retry_api"          -> Action.RETRY_API
            "compression_done"   -> Action.CONTINUE_GENERATION (from COMPRESSING)
        """
        def _do_mirror():
            if not value:
                return None  # Only act on flag being set to True / truthy

            old_state = self.sm.state

            if flag_name == "messages_received":
                messages = extra_context.get("messages", [{"role": "user", "content": ""}])
                system_prompt = extra_context.get("system_prompt")
                self.sm.receive_message(messages, system_prompt)

            elif flag_name == "api_call_started":
                api_params = extra_context.get("api_params", {"messages": []})
                self.sm.build_request(api_params)

            elif flag_name == "response_received":
                response = extra_context.get("response", {})
                finish_reason = extra_context.get("finish_reason", "stop")
                tool_calls = extra_context.get("tool_calls")
                token_count = extra_context.get("token_count")
                self.sm.receive_response(
                    response, finish_reason, tool_calls, token_count
                )

            elif flag_name == "api_error":
                error = extra_context.get("error", Exception("unknown"))
                self.sm.recover_error(error)

            elif flag_name == "tool_dispatch":
                self.sm.dispatch_tools()

            elif flag_name == "tools_complete":
                results = extra_context.get("results", [])
                self.sm.tool_complete(results)

            elif flag_name == "text_response":
                self.sm.process_text_response()

            elif flag_name == "restart_with_compressed_messages":
                compressed = extra_context.get("compressed_messages", [])
                token_count = extra_context.get("token_count", 0)
                self.sm.compress(compressed, token_count)

            elif flag_name == "compression_done":
                self.sm.continue_after_compression()

            elif flag_name == "restart_with_length_continuation":
                self.sm.continue_generation()

            elif flag_name == "budget_exceeded":
                self.sm.exhaust_budget()

            elif flag_name == "interrupted":
                self.sm.interrupt()

            elif flag_name == "return_result":
                self.sm.return_result()

            elif flag_name == "retry_api":
                self.sm.retry_api()

            else:
                logger.debug("Shadow SM: unknown flag '%s'", flag_name)
                return None

            new_state = self.sm.state
            self.log_transition(flag_name, old_state, new_state)
            return new_state.value

        return self._safe("mirror_flag_change", _do_mirror)

    def mirror_phase(
        self,
        phase: str,
        **extra_context,
    ) -> Optional[str]:
        """Mirror a high-level phase of the agent loop.

        This is a convenience method for code points in run_agent.py that
        don't correspond to a single flag change but to a phase boundary.

        Supported phases:
            "loop_start"        -> ensure in AWAITING_INPUT
            "preparing_request" -> ensure in PREPARING_API_CALL
            "api_call_start"    -> ensure in CALLING_API
            "response_received" -> ensure in PROCESSING_RESPONSE
            "tool_execution"    -> ensure in EXECUTING_TOOLS
            "continuation_check"-> ensure in HANDLING_CONTINUATION
            "loop_exit"         -> ensure in terminal state
        """
        # Phase mirroring is just sugar over mirror_flag_change
        phase_to_flag = {
            "loop_start": "messages_received",
            "api_call_start": "api_call_started",
            "response_received": "response_received",
            "tool_execution": "tool_dispatch",
            "tools_done": "tools_complete",
            "text_response": "text_response",
            "loop_exit": "return_result",
        }
        flag = phase_to_flag.get(phase)
        if flag:
            return self.mirror_flag_change(flag, True, **extra_context)
        return None

    # ---- Divergence checking ----

    def check_divergence(
        self,
        current_flags: Dict[str, Any],
        context: str = "",
    ) -> Optional[str]:
        """Compare the SM state to the boolean flags and return a divergence message.

        Returns None if states match, or a description string if they diverge.
        """
        def _do_check():
            expected = _derive_expected_state(current_flags)
            actual = self.sm.state

            # Allow some fuzzy matching for intermediate states
            # The SM may be one step ahead or behind the flags
            compatible = _states_compatible(actual, expected, current_flags)

            if not compatible:
                record = DivergenceRecord(
                    timestamp=time.time(),
                    sm_state=actual.value,
                    expected_state=expected.value,
                    flags={k: v for k, v in current_flags.items() if v},
                    context=context,
                )
                self.divergences.append(record)
                msg = str(record)
                logger.warning("Shadow SM divergence: %s", msg)
                return msg

            return None

        return self._safe("check_divergence", _do_check)

    # ---- Invariant checking ----

    def check_invariants(self) -> List[str]:
        """Run all invariant checks on the shadow SM.

        Returns a list of violation messages (empty if all pass).
        """
        def _do_check():
            violations = self.sm.check_invariants()
            if violations:
                record = {
                    "timestamp": time.time(),
                    "state": self.sm.state.value,
                    "violations": violations,
                }
                self._invariant_violations.append(record)
                for v in violations:
                    logger.warning("Shadow SM invariant violation: %s", v)
            return violations

        result = self._safe("check_invariants", _do_check)
        return result if result is not None else []

    # ---- Transition logging ----

    def log_transition(
        self,
        action: str,
        old_state: AgentLoopState,
        new_state: AgentLoopState,
    ) -> None:
        """Log a transition with structured data."""
        self._transition_count += 1
        entry = {
            "seq": self._transition_count,
            "timestamp": time.time(),
            "action": action,
            "from_state": old_state.value,
            "to_state": new_state.value,
        }
        self.transition_log.append(entry)
        logger.debug(
            "Shadow SM transition #%d: %s -> %s (via %s)",
            self._transition_count,
            old_state.value,
            new_state.value,
            action,
        )

    # ---- Force state (for recovery / testing) ----

    def force_state(self, state: AgentLoopState, reason: str = "") -> None:
        """Force the shadow SM into a specific state."""
        self._safe(
            "force_state",
            lambda: self.sm.force_state(state, reason),
        )

    def reset(self) -> None:
        """Reset the shadow SM for a new turn."""
        self._safe("reset", lambda: self.sm.reset())

    # ---- Introspection ----

    @property
    def state(self) -> Optional[str]:
        """Current shadow SM state as a string, or None if disabled."""
        if not self.enabled or self.sm is None:
            return None
        try:
            return self.sm.state.value
        except Exception:
            return None

    @property
    def divergence_count(self) -> int:
        """Total number of divergences detected."""
        return len(self.divergences)

    @property
    def invariant_violation_count(self) -> int:
        """Total number of invariant violation batches."""
        return len(self._invariant_violations)

    @property
    def error_count(self) -> int:
        """Total number of internal shadow errors."""
        return self._error_count

    @property
    def transition_count(self) -> int:
        """Total number of transitions recorded."""
        return self._transition_count

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for telemetry / debugging."""
        return {
            "enabled": self.enabled,
            "state": self.state,
            "transitions": self._transition_count,
            "divergences": self.divergence_count,
            "invariant_violations": self.invariant_violation_count,
            "errors": self._error_count,
        }

    def __repr__(self) -> str:
        if not self.enabled:
            return "AgentLoopShadow(disabled)"
        return (
            f"AgentLoopShadow("
            f"state={self.state}, "
            f"transitions={self._transition_count}, "
            f"divergences={self.divergence_count}, "
            f"errors={self._error_count})"
        )


# ---------------------------------------------------------------------------
# Compatibility heuristics
# ---------------------------------------------------------------------------

def _states_compatible(
    sm_state: AgentLoopState,
    expected: AgentLoopState,
    flags: Dict[str, Any],
) -> bool:
    """Check if SM state and derived-expected state are compatible.

    Exact match is always compatible. We also allow some fuzzy matches
    for states that are "adjacent" in the transition graph, since the
    shadow may be one step ahead or behind the flags due to timing.
    """
    if sm_state == expected:
        return True

    # The SM may have already moved to PREPARING_API_CALL while flags
    # still say HANDLING_CONTINUATION or COMPRESSING_CONTEXT
    adjacent_pairs = {
        # (sm_state, expected) pairs that are OK
        (AgentLoopState.PREPARING_API_CALL, AgentLoopState.HANDLING_CONTINUATION),
        (AgentLoopState.PREPARING_API_CALL, AgentLoopState.COMPRESSING_CONTEXT),
        (AgentLoopState.PREPARING_API_CALL, AgentLoopState.AWAITING_INPUT),
        (AgentLoopState.HANDLING_CONTINUATION, AgentLoopState.PROCESSING_RESPONSE),
        (AgentLoopState.HANDLING_CONTINUATION, AgentLoopState.EXECUTING_TOOLS),
        (AgentLoopState.CALLING_API, AgentLoopState.PREPARING_API_CALL),
        # Terminal states are compatible with each other in edge cases
        (AgentLoopState.RETURNING_RESPONSE, AgentLoopState.BUDGET_EXHAUSTED),
        (AgentLoopState.BUDGET_EXHAUSTED, AgentLoopState.RETURNING_RESPONSE),
    }

    return (sm_state, expected) in adjacent_pairs

