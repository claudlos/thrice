"""
Agent Loop State Machine (SM-2) for Hermes.

Extracts the implicit boolean-flag-driven control flow from run_agent.py's
~1700-line while loop into an explicit state machine with enum states,
guarded transitions, and runtime invariant checking.

Implements the formal specification from:
    specs/SM-2-agent-loop.md
    specs/tla/AgentLoop.tla

State/flag mapping (old -> new):
    restart_with_compressed_messages = True  ->  COMPRESSING_CONTEXT
    restart_with_length_continuation = True  ->  HANDLING_CONTINUATION
    budget_exceeded = True                   ->  BUDGET_EXHAUSTED
    interrupted = True                       ->  INTERRUPTED
    (normal entry)                           ->  AWAITING_INPUT
    (building params)                        ->  PREPARING_API_CALL
    (await API response)                     ->  CALLING_API
    (parsing response)                       ->  PROCESSING_RESPONSE
    (running tool calls)                     ->  EXECUTING_TOOLS
    (returning to caller)                    ->  RETURNING_RESPONSE
    (in except block with retries)           ->  ERROR_RECOVERY

Usage:
    from agent_loop_state_machine import AgentLoopStateMachine, AgentLoopState

    sm = AgentLoopStateMachine()
    sm.receive_message(messages=[...])
    sm.build_request()
    sm.send_api_call()
    # ... etc.
"""

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from state_machine import (
    ANY_STATE,
    StateMachine,
    TransitionDef,
    TransitionRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent loop states
# ---------------------------------------------------------------------------

class AgentLoopState(enum.Enum):
    """Explicit states for the agent loop.

    Each state corresponds to a well-defined phase of the agent's
    request-response cycle.  The old code encoded these as combinations
    of boolean flags; the mapping is documented in each member's comment.
    """

    # Waiting for user/caller to provide messages.
    # Old: top of while loop, before any flags are set
    AWAITING_INPUT = "awaiting_input"

    # Constructing the API request (system prompt, params, etc.).
    # Old: section that builds api_params dict
    PREPARING_API_CALL = "preparing_api_call"

    # Blocked on the API HTTP call.
    # Old: inside `await client.messages.create(...)` or streaming
    CALLING_API = "calling_api"

    # Parsing the API response (extract text, tool_calls, finish_reason).
    # Old: section after response is received, before tool dispatch
    PROCESSING_RESPONSE = "processing_response"

    # Running tool calls returned by the model.
    # Old: `for tool_call in tool_calls:` loop
    EXECUTING_TOOLS = "executing_tools"

    # Summarizing / truncating the message list because it's too large.
    # Old: restart_with_compressed_messages = True
    COMPRESSING_CONTEXT = "compressing_context"

    # Deciding whether to continue (finish_reason=length) or return.
    # Old: restart_with_length_continuation = True / post-tool logic
    HANDLING_CONTINUATION = "handling_continuation"

    # Done: returning the final assistant response to the caller.
    # Old: break out of while loop, return result
    RETURNING_RESPONSE = "returning_response"

    # Ctrl-C or SIGINT received.
    # Old: interrupted = True
    INTERRUPTED = "interrupted"

    # Iteration or token budget exceeded.
    # Old: budget_exceeded = True
    BUDGET_EXHAUSTED = "budget_exhausted"

    # API call failed; deciding whether to retry.
    # Old: inside except block checking retry_count
    ERROR_RECOVERY = "error_recovery"


# Convenience sets
_TERMINAL_STATES = frozenset({
    AgentLoopState.RETURNING_RESPONSE.value,
    AgentLoopState.BUDGET_EXHAUSTED.value,
    AgentLoopState.INTERRUPTED.value,
})

_ALL_STATE_VALUES = frozenset(s.value for s in AgentLoopState)

# States from which interrupt can fire (everything except terminal + interrupted)
_INTERRUPTIBLE_STATES = _ALL_STATE_VALUES - _TERMINAL_STATES - {
    AgentLoopState.INTERRUPTED.value,
}


# ---------------------------------------------------------------------------
# Agent loop context (replaces scattered boolean flags)
# ---------------------------------------------------------------------------

@dataclass
class AgentLoopContext:
    """Structured context that travels with the state machine.

    Replaces the implicit locals scattered across the while loop.
    """

    # Messages
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None

    # Budget tracking
    iteration_budget: int = 25
    iterations_used: int = 0
    token_count: int = 0
    max_tokens: int = 200_000
    compression_threshold: int = 150_000

    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[Exception] = None

    # Response tracking
    last_response: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Interrupt
    interrupt_flag: bool = False

    # API params (built during PREPARING_API_CALL)
    api_params: Optional[Dict[str, Any]] = None

    # Force continuation (e.g., tool follow-up)
    force_continuation: bool = False

    @property
    def has_budget(self) -> bool:
        return (
            self.iterations_used < self.iteration_budget
            and self.token_count < self.max_tokens
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.pending_tool_calls)

    @property
    def needs_continue(self) -> bool:
        return (
            self.finish_reason == "length"
            or self.force_continuation
        )

    @property
    def context_full(self) -> bool:
        return self.token_count > self.compression_threshold

    @property
    def turn_done(self) -> bool:
        return (
            self.finish_reason == "stop"
            and not self.needs_continue
            and not self.context_full
        )

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def reset_retry(self) -> None:
        """Reset retry count after a successful API call (INV-A5)."""
        self.retry_count = 0
        self.last_error = None

    def increment_iteration(self) -> None:
        """Consume one iteration of the budget."""
        self.iterations_used += 1

    def to_guard_dict(self) -> Dict[str, Any]:
        """Export as a flat dict for guard function consumption."""
        return {
            "messages": self.messages,
            "has_budget": self.has_budget,
            "has_tool_calls": self.has_tool_calls,
            "needs_continue": self.needs_continue,
            "context_full": self.context_full,
            "turn_done": self.turn_done,
            "can_retry": self.can_retry,
            "interrupt_flag": self.interrupt_flag,
            "iterations_used": self.iterations_used,
            "iteration_budget": self.iteration_budget,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "token_count": self.token_count,
            "finish_reason": self.finish_reason,
            "api_params": self.api_params,
        }


# ---------------------------------------------------------------------------
# Guard functions
# ---------------------------------------------------------------------------

def _guard_messages_non_empty(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Messages list must be non-empty to start processing."""
    return bool(ctx.get("messages"))


def _guard_has_budget(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Budget must remain for another iteration."""
    return ctx.get("has_budget", False)


def _guard_no_budget(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Budget is exhausted."""
    return not ctx.get("has_budget", True)


def _guard_api_params_valid(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """API params must have been built."""
    return ctx.get("api_params") is not None


def _guard_has_tools(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Response contains tool calls."""
    return ctx.get("has_tool_calls", False)


def _guard_no_tools(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Response has no tool calls."""
    return not ctx.get("has_tool_calls", False)


def _guard_needs_continue(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Model output was truncated (finish_reason=length) or needs follow-up."""
    return ctx.get("needs_continue", False)


def _guard_context_full(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Context window is too large, needs compression."""
    return ctx.get("context_full", False)


def _guard_turn_done(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """The turn is complete (stop, no continuation, no compression needed)."""
    return ctx.get("turn_done", False)


def _guard_can_retry(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Retry budget allows another attempt."""
    return ctx.get("can_retry", False)


def _guard_retry_exhausted(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Retry budget is spent."""
    return not ctx.get("can_retry", True)


def _guard_interruptible(
    current_state: str, action: str, ctx: Dict[str, Any]
) -> bool:
    """Current state allows interruption."""
    return current_state in _INTERRUPTIBLE_STATES


# ---------------------------------------------------------------------------
# Invariant functions
# ---------------------------------------------------------------------------

def _inv_messages_non_empty_before_call(sm: StateMachine) -> List[str]:
    """INV-A3: Messages must be non-empty when in CALLING_API."""
    if sm.state == AgentLoopState.CALLING_API.value:
        ctx = sm.context.get("loop_ctx")
        if ctx and not ctx.messages:
            return ["INV-A3: Messages empty in CALLING_API state"]
    return []


def _inv_budget_monotonic(sm: StateMachine) -> List[str]:
    """INV-A4: iterations_used never exceeds budget (within tolerance)."""
    ctx = sm.context.get("loop_ctx")
    if ctx and ctx.iterations_used > ctx.iteration_budget + 1:
        return [
            f"INV-A4: iterations_used ({ctx.iterations_used}) "
            f"exceeds budget ({ctx.iteration_budget}) by more than 1"
        ]
    return []


def _inv_retry_bounded(sm: StateMachine) -> List[str]:
    """INV-A5 related: retry_count must not exceed max_retries."""
    ctx = sm.context.get("loop_ctx")
    if ctx and ctx.retry_count > ctx.max_retries:
        return [
            f"INV-A5: retry_count ({ctx.retry_count}) "
            f"exceeds max_retries ({ctx.max_retries})"
        ]
    return []


def _inv_retry_resets_on_success(sm: StateMachine) -> List[str]:
    """INV-A5: retry_count resets to 0 after successful API response."""
    if sm.state == AgentLoopState.PROCESSING_RESPONSE.value:
        ctx = sm.context.get("loop_ctx")
        if ctx and ctx.retry_count != 0:
            return [
                f"INV-A5: retry_count should be 0 in PROCESSING_RESPONSE "
                f"but is {ctx.retry_count}"
            ]
    return []


def _inv_system_prompt_preserved(sm: StateMachine) -> List[str]:
    """INV-A8: System prompt (first message) is never removed."""
    ctx = sm.context.get("loop_ctx")
    if ctx and ctx.messages and ctx.system_prompt:
        first = ctx.messages[0] if ctx.messages else None
        if first and first.get("role") != "system":
            return ["INV-A8: System prompt missing from messages[0]"]
    return []


# ---------------------------------------------------------------------------
# Actions (string constants matching the spec)
# ---------------------------------------------------------------------------

class Action:
    """Named actions corresponding to the transition function in SM-2."""
    RECEIVE_MESSAGE = "receive_message"
    BUILD_REQUEST = "build_request"
    SEND_API_CALL = "send_api_call"
    RECEIVE_RESPONSE = "receive_response"
    DISPATCH_TOOLS = "dispatch_tools"
    TOOL_COMPLETE = "tool_complete"
    COMPRESS = "compress"
    CONTINUE_GENERATION = "continue_generation"
    RETURN_RESULT = "return_result"
    INTERRUPT = "interrupt"
    EXHAUST_BUDGET = "exhaust_budget"
    RECOVER_ERROR = "recover_error"
    RETRY_API = "retry_api"


# ---------------------------------------------------------------------------
# AgentLoopStateMachine
# ---------------------------------------------------------------------------

class AgentLoopStateMachine:
    """State machine for the Hermes agent loop.

    Wraps the generic StateMachine with agent-loop-specific states,
    transitions, guards, invariants, and a typed context.

    Corresponds 1:1 to the TLA+ spec in specs/tla/AgentLoop.tla
    and the formal definition in specs/SM-2-agent-loop.md.
    """

    def __init__(
        self,
        iteration_budget: int = 25,
        max_retries: int = 3,
        max_tokens: int = 200_000,
        compression_threshold: int = 150_000,
        check_invariants: bool = True,
    ):
        self._loop_ctx = AgentLoopContext(
            iteration_budget=iteration_budget,
            max_retries=max_retries,
            max_tokens=max_tokens,
            compression_threshold=compression_threshold,
        )

        S = AgentLoopState
        A = Action

        # -- Build transition table matching SM-2 spec --
        transitions = {
            A.RECEIVE_MESSAGE: [
                TransitionDef(
                    S.AWAITING_INPUT.value,
                    S.PREPARING_API_CALL.value,
                    guard=_guard_messages_non_empty,
                    guard_name="messages_non_empty",
                ),
            ],
            A.BUILD_REQUEST: [
                TransitionDef(
                    S.PREPARING_API_CALL.value,
                    S.CALLING_API.value,
                    guard=_guard_has_budget,
                    guard_name="has_budget",
                ),
                # Also from COMPRESSING_CONTEXT (after compression)
                TransitionDef(
                    S.COMPRESSING_CONTEXT.value,
                    S.PREPARING_API_CALL.value,
                    guard_name="compression_done",
                ),
            ],
            A.EXHAUST_BUDGET: [
                TransitionDef(
                    S.PREPARING_API_CALL.value,
                    S.BUDGET_EXHAUSTED.value,
                    guard=_guard_no_budget,
                    guard_name="no_budget",
                ),
                TransitionDef(
                    S.HANDLING_CONTINUATION.value,
                    S.BUDGET_EXHAUSTED.value,
                    guard=_guard_no_budget,
                    guard_name="no_budget",
                ),
            ],
            A.RECEIVE_RESPONSE: [
                TransitionDef(
                    S.CALLING_API.value,
                    S.PROCESSING_RESPONSE.value,
                    guard_name="response_received",
                ),
            ],
            A.RECOVER_ERROR: [
                TransitionDef(
                    S.CALLING_API.value,
                    S.ERROR_RECOVERY.value,
                    guard_name="api_error",
                ),
            ],
            A.DISPATCH_TOOLS: [
                TransitionDef(
                    S.PROCESSING_RESPONSE.value,
                    S.EXECUTING_TOOLS.value,
                    guard=_guard_has_tools,
                    guard_name="has_tools",
                ),
            ],
            A.RETURN_RESULT: [
                # From PROCESSING_RESPONSE (text-only, no continuation)
                TransitionDef(
                    S.HANDLING_CONTINUATION.value,
                    S.RETURNING_RESPONSE.value,
                    guard=_guard_turn_done,
                    guard_name="turn_done",
                ),
                # From ERROR_RECOVERY (retries exhausted)
                TransitionDef(
                    S.ERROR_RECOVERY.value,
                    S.RETURNING_RESPONSE.value,
                    guard=_guard_retry_exhausted,
                    guard_name="retry_exhausted",
                ),
            ],
            A.TOOL_COMPLETE: [
                TransitionDef(
                    S.EXECUTING_TOOLS.value,
                    S.HANDLING_CONTINUATION.value,
                    guard_name="tools_executed",
                ),
            ],
            A.COMPRESS: [
                TransitionDef(
                    S.HANDLING_CONTINUATION.value,
                    S.COMPRESSING_CONTEXT.value,
                    guard=_guard_context_full,
                    guard_name="context_full",
                ),
            ],
            A.CONTINUE_GENERATION: [
                TransitionDef(
                    S.HANDLING_CONTINUATION.value,
                    S.PREPARING_API_CALL.value,
                    guard=_guard_needs_continue,
                    guard_name="needs_continue",
                ),
                # After compression
                TransitionDef(
                    S.COMPRESSING_CONTEXT.value,
                    S.PREPARING_API_CALL.value,
                    guard_name="compression_complete",
                ),
            ],
            A.RETRY_API: [
                TransitionDef(
                    S.ERROR_RECOVERY.value,
                    S.CALLING_API.value,
                    guard=_guard_can_retry,
                    guard_name="can_retry",
                ),
            ],
            # Text-only response goes to HANDLING_CONTINUATION
            # (matches TLA+ ProcessTextResponse)
            "process_text_response": [
                TransitionDef(
                    S.PROCESSING_RESPONSE.value,
                    S.HANDLING_CONTINUATION.value,
                    guard=_guard_no_tools,
                    guard_name="no_tools",
                ),
            ],
            A.INTERRUPT: [
                # Wildcard: any interruptible state -> INTERRUPTED
                TransitionDef(
                    ANY_STATE,
                    S.INTERRUPTED.value,
                    guard=_guard_interruptible,
                    guard_name="interruptible",
                ),
            ],
        }

        invariants = [
            _inv_messages_non_empty_before_call,
            _inv_budget_monotonic,
            _inv_retry_bounded,
            _inv_retry_resets_on_success,
            _inv_system_prompt_preserved,
        ]

        self._sm = StateMachine(
            name="agent_loop",
            states=_ALL_STATE_VALUES,
            initial_state=S.AWAITING_INPUT.value,
            transitions=transitions,
            invariants=invariants,
            check_invariants_on_transition=check_invariants,
        )

        # Store the loop context in the state machine's context dict
        self._sm.context["loop_ctx"] = self._loop_ctx

    # ---- Properties ----

    @property
    def state(self) -> AgentLoopState:
        """Current state as an enum member."""
        return AgentLoopState(self._sm.state)

    @property
    def state_value(self) -> str:
        """Current state as a string."""
        return self._sm.state

    @property
    def loop_context(self) -> AgentLoopContext:
        """The typed agent loop context."""
        return self._loop_ctx

    @property
    def history(self) -> List[TransitionRecord]:
        """Transition history."""
        return self._sm.history

    @property
    def is_terminal(self) -> bool:
        """Whether the state machine is in a terminal state."""
        return self._sm.state in _TERMINAL_STATES

    @property
    def inner(self) -> StateMachine:
        """Access the underlying StateMachine (for advanced use)."""
        return self._sm

    # ---- Guard-context helper ----

    def _ctx(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the guard context dict from current AgentLoopContext."""
        d = self._loop_ctx.to_guard_dict()
        if extra:
            d.update(extra)
        return d

    # ---- High-level transition methods ----
    # Each method corresponds to an action in the SM-2 spec.
    # They update the AgentLoopContext AND apply the state transition.

    def receive_message(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> TransitionRecord:
        """Caller provides messages to begin a turn.

        AWAITING_INPUT -> PREPARING_API_CALL
        Guard: messages non-empty
        """
        self._loop_ctx.messages = messages
        if system_prompt is not None:
            self._loop_ctx.system_prompt = system_prompt
        return self._sm.apply(Action.RECEIVE_MESSAGE, self._ctx())

    def build_request(self, api_params: Dict[str, Any]) -> TransitionRecord:
        """Build the API request parameters.

        PREPARING_API_CALL -> CALLING_API  (when has_budget)
        Guard: has_budget

        The iteration counter is incremented *after* the guard check so the
        guard sees the pre-increment value (matches TLA+ ``BuildRequest`` in
        ``specs/tla/AgentLoop.tla`` where ``iterationsUsed' = iterationsUsed
        + 1`` is atomic with the transition).
        """
        self._loop_ctx.api_params = api_params
        record = self._sm.apply(Action.BUILD_REQUEST, self._ctx())
        self._loop_ctx.increment_iteration()
        return record

    def exhaust_budget(self) -> TransitionRecord:
        """Signal that the budget is exhausted.

        PREPARING_API_CALL | HANDLING_CONTINUATION -> BUDGET_EXHAUSTED
        Guard: no budget remaining
        """
        return self._sm.apply(Action.EXHAUST_BUDGET, self._ctx())

    def receive_response(
        self,
        response: Dict[str, Any],
        finish_reason: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        token_count: Optional[int] = None,
    ) -> TransitionRecord:
        """API response received successfully.

        CALLING_API -> PROCESSING_RESPONSE
        Side effect: resets retry_count (INV-A5)
        """
        self._loop_ctx.last_response = response
        self._loop_ctx.finish_reason = finish_reason
        self._loop_ctx.pending_tool_calls = tool_calls or []
        self._loop_ctx.reset_retry()  # INV-A5
        if token_count is not None:
            self._loop_ctx.token_count = token_count
        return self._sm.apply(Action.RECEIVE_RESPONSE, self._ctx())

    def recover_error(self, error: Exception) -> TransitionRecord:
        """API call failed.

        CALLING_API -> ERROR_RECOVERY
        """
        self._loop_ctx.last_error = error
        return self._sm.apply(Action.RECOVER_ERROR, self._ctx())

    def dispatch_tools(self) -> TransitionRecord:
        """Begin executing tool calls.

        PROCESSING_RESPONSE -> EXECUTING_TOOLS
        Guard: has_tool_calls
        """
        return self._sm.apply(Action.DISPATCH_TOOLS, self._ctx())

    def process_text_response(self) -> TransitionRecord:
        """Text-only response, move to continuation handling.

        PROCESSING_RESPONSE -> HANDLING_CONTINUATION
        Guard: no tool_calls
        """
        return self._sm.apply("process_text_response", self._ctx())

    def tool_complete(
        self, results: List[Dict[str, Any]]
    ) -> TransitionRecord:
        """All tool calls have completed.

        EXECUTING_TOOLS -> HANDLING_CONTINUATION
        """
        self._loop_ctx.tool_results = results
        self._loop_ctx.pending_tool_calls = []
        # Append tool result messages
        for result in results:
            self._loop_ctx.messages.append(result)
        return self._sm.apply(Action.TOOL_COMPLETE, self._ctx())

    def compress(
        self, compressed_messages: List[Dict[str, Any]], new_token_count: int
    ) -> TransitionRecord:
        """Compress context and loop back to preparing API call.

        HANDLING_CONTINUATION -> COMPRESSING_CONTEXT
        Guard: context_full
        """
        record = self._sm.apply(Action.COMPRESS, self._ctx())
        # Apply compression
        self._loop_ctx.messages = compressed_messages
        self._loop_ctx.token_count = new_token_count
        return record

    def continue_after_compression(self) -> TransitionRecord:
        """After compression, continue to PREPARING_API_CALL.

        COMPRESSING_CONTEXT -> PREPARING_API_CALL
        """
        return self._sm.apply(Action.CONTINUE_GENERATION, self._ctx())

    def continue_generation(self) -> TransitionRecord:
        """Continue generation (finish_reason=length or force).

        HANDLING_CONTINUATION -> PREPARING_API_CALL
        Guard: needs_continue
        """
        return self._sm.apply(Action.CONTINUE_GENERATION, self._ctx())

    def return_result(self) -> TransitionRecord:
        """Return the final response.

        HANDLING_CONTINUATION -> RETURNING_RESPONSE  (when turn_done)
        ERROR_RECOVERY -> RETURNING_RESPONSE  (when retry_exhausted)
        """
        return self._sm.apply(Action.RETURN_RESULT, self._ctx())

    def retry_api(self) -> TransitionRecord:
        """Retry the API call after an error.

        ERROR_RECOVERY -> CALLING_API
        Guard: can_retry

        The generic ``StateMachine.apply`` raises on guard failure, so
        reaching the line after it always means the transition was
        accepted.  Matches TLA+ ``RetryAPI`` in ``specs/tla/AgentLoop.tla``
        where ``retryCount' = retryCount + 1`` happens unconditionally
        with the transition.
        """
        result = self._sm.apply(Action.RETRY_API, self._ctx())
        self._loop_ctx.retry_count += 1
        return result

    def interrupt(self) -> TransitionRecord:
        """Signal an interrupt (Ctrl-C / SIGINT).

        ANY (interruptible) -> INTERRUPTED
        """
        self._loop_ctx.interrupt_flag = True
        return self._sm.apply(Action.INTERRUPT, self._ctx())

    # ---- Decision helpers ----

    def decide_after_response(self) -> str:
        """Determine the correct action after receiving a response.

        Returns the action name to call next based on the current context.
        This replaces the if/elif chain in the old while loop.
        """
        ctx = self._loop_ctx
        if ctx.has_tool_calls:
            return Action.DISPATCH_TOOLS
        else:
            return "process_text_response"

    def decide_after_continuation(self) -> str:
        """Determine the correct action in HANDLING_CONTINUATION.

        Returns the action name to call next. Replaces the old:
            if budget_exceeded: ...
            elif restart_with_compressed_messages: ...
            elif restart_with_length_continuation: ...
            else: return result
        """
        ctx = self._loop_ctx
        if not ctx.has_budget:
            return Action.EXHAUST_BUDGET
        if ctx.context_full:
            return Action.COMPRESS
        if ctx.needs_continue:
            return Action.CONTINUE_GENERATION
        return Action.RETURN_RESULT

    def decide_after_error(self) -> str:
        """Determine the correct action in ERROR_RECOVERY."""
        if self._loop_ctx.can_retry:
            return Action.RETRY_API
        return Action.RETURN_RESULT

    # ---- Introspection ----

    def available_actions(self) -> List[str]:
        """List actions valid from the current state."""
        return self._sm.get_available_actions()

    def check_invariants(self) -> List[str]:
        """Run all invariant checks and return violations."""
        return self._sm.check_invariants()

    def force_state(self, state: AgentLoopState, reason: str = "") -> None:
        """Force the machine into a specific state (testing/recovery)."""
        self._sm.force_state(state.value, reason)

    def reset(self) -> None:
        """Reset to initial state for a new turn."""
        self._sm.force_state(
            AgentLoopState.AWAITING_INPUT.value, reason="reset"
        )
        self._loop_ctx.last_response = None
        self._loop_ctx.finish_reason = None
        self._loop_ctx.pending_tool_calls = []
        self._loop_ctx.tool_results = []
        self._loop_ctx.api_params = None
        self._loop_ctx.last_error = None
        self._loop_ctx.force_continuation = False
        # Note: messages, iterations_used, token_count are preserved
        # across turns (they accumulate). retry_count is preserved too.

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for debugging."""
        return {
            "state": self.state.value,
            "is_terminal": self.is_terminal,
            "iterations_used": self._loop_ctx.iterations_used,
            "iteration_budget": self._loop_ctx.iteration_budget,
            "retry_count": self._loop_ctx.retry_count,
            "token_count": self._loop_ctx.token_count,
            "message_count": len(self._loop_ctx.messages),
            "finish_reason": self._loop_ctx.finish_reason,
            "interrupt_flag": self._loop_ctx.interrupt_flag,
            "available_actions": self.available_actions(),
            "history_length": len(self._sm.history),
        }

    def __repr__(self) -> str:
        return (
            f"AgentLoopStateMachine("
            f"state={self.state.value}, "
            f"iter={self._loop_ctx.iterations_used}/{self._loop_ctx.iteration_budget}, "
            f"retry={self._loop_ctx.retry_count}/{self._loop_ctx.max_retries})"
        )
