"""
Tests for AgentLoopShadow (Phase 1: shadow mode).

Tests cover:
- Initialization and basic properties
- Flag mirroring to SM transitions
- Divergence detection
- Invariant checking
- Error resilience (shadow never crashes)
- _derive_expected_state helper
- _states_compatible heuristic
- Full loop simulation
"""

import os
import sys

# Ensure the new-files directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from agent_loop_shadow import (
    AgentLoopShadow,
    DivergenceRecord,
    _derive_expected_state,
    _states_compatible,
)
from agent_loop_state_machine import (
    AgentLoopState,
)

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestShadowInit:
    def test_default_init(self):
        shadow = AgentLoopShadow()
        assert shadow.enabled is True
        assert shadow.sm is not None
        assert shadow.state == AgentLoopState.AWAITING_INPUT.value
        assert shadow.divergence_count == 0
        assert shadow.error_count == 0
        assert shadow.transition_count == 0

    def test_disabled_init(self):
        shadow = AgentLoopShadow(enabled=False)
        assert shadow.enabled is False
        assert shadow.state is None

    def test_custom_budget(self):
        shadow = AgentLoopShadow(iteration_budget=10, max_retries=5)
        assert shadow.sm.loop_context.iteration_budget == 10
        assert shadow.sm.loop_context.max_retries == 5

    def test_repr_enabled(self):
        shadow = AgentLoopShadow()
        r = repr(shadow)
        assert "AgentLoopShadow(" in r
        assert "state=" in r

    def test_repr_disabled(self):
        shadow = AgentLoopShadow(enabled=False)
        assert repr(shadow) == "AgentLoopShadow(disabled)"


# ---------------------------------------------------------------------------
# Flag mirroring
# ---------------------------------------------------------------------------

class TestMirrorFlagChange:
    def test_messages_received(self):
        shadow = AgentLoopShadow()
        result = shadow.mirror_flag_change(
            "messages_received", True,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result == AgentLoopState.PREPARING_API_CALL.value
        assert shadow.state == AgentLoopState.PREPARING_API_CALL.value

    def test_false_value_does_nothing(self):
        shadow = AgentLoopShadow()
        result = shadow.mirror_flag_change("interrupted", False)
        assert result is None
        assert shadow.state == AgentLoopState.AWAITING_INPUT.value

    def test_api_call_started(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change(
            "messages_received", True,
            messages=[{"role": "user", "content": "hi"}],
        )
        result = shadow.mirror_flag_change(
            "api_call_started", True,
            api_params={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert result == AgentLoopState.CALLING_API.value

    def test_response_received(self):
        shadow = AgentLoopShadow()
        # Walk through to CALLING_API
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})
        result = shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        assert result == AgentLoopState.PROCESSING_RESPONSE.value

    def test_tool_dispatch(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})
        # Set up tool calls in context
        shadow.sm.loop_context.pending_tool_calls = [{"id": "1", "name": "test"}]
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop",
            tool_calls=[{"id": "1", "name": "test"}])
        result = shadow.mirror_flag_change("tool_dispatch", True)
        assert result == AgentLoopState.EXECUTING_TOOLS.value

    def test_text_response(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        result = shadow.mirror_flag_change("text_response", True)
        assert result == AgentLoopState.HANDLING_CONTINUATION.value

    def test_interrupt(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        result = shadow.mirror_flag_change("interrupted", True)
        assert result == AgentLoopState.INTERRUPTED.value

    def test_return_result(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        shadow.mirror_flag_change("text_response", True)
        # Set turn_done
        shadow.sm.loop_context.finish_reason = "stop"
        shadow.sm.loop_context.force_continuation = False
        result = shadow.mirror_flag_change("return_result", True)
        assert result == AgentLoopState.RETURNING_RESPONSE.value

    def test_unknown_flag(self):
        shadow = AgentLoopShadow()
        result = shadow.mirror_flag_change("nonexistent_flag", True)
        assert result is None

    def test_transition_log_populated(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        assert shadow.transition_count == 1
        assert len(shadow.transition_log) == 1
        entry = shadow.transition_log[0]
        assert entry["action"] == "messages_received"
        assert entry["from_state"] == AgentLoopState.AWAITING_INPUT.value
        assert entry["to_state"] == AgentLoopState.PREPARING_API_CALL.value


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

class TestDivergenceDetection:
    def test_no_divergence_matching_states(self):
        shadow = AgentLoopShadow()
        # SM is in AWAITING_INPUT, flags also indicate AWAITING_INPUT
        result = shadow.check_divergence({
            "interrupted": False,
            "budget_exceeded": False,
            "restart_with_compressed_messages": False,
            "restart_with_length_continuation": False,
        })
        assert result is None
        assert shadow.divergence_count == 0

    def test_divergence_detected(self):
        shadow = AgentLoopShadow()
        # SM is in AWAITING_INPUT, but flags say interrupted
        result = shadow.check_divergence({
            "interrupted": True,
        })
        assert result is not None
        assert "interrupted" in result
        assert shadow.divergence_count == 1

    def test_compatible_states_no_divergence(self):
        shadow = AgentLoopShadow()
        # Move SM to PREPARING_API_CALL
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        # Flags indicate AWAITING_INPUT (compatible adjacent state)
        result = shadow.check_divergence({
            "interrupted": False,
            "budget_exceeded": False,
            "restart_with_compressed_messages": False,
            "restart_with_length_continuation": False,
        })
        assert result is None

    def test_divergence_record_fields(self):
        record = DivergenceRecord(
            timestamp=1000.0,
            sm_state="awaiting_input",
            expected_state="interrupted",
            flags={"interrupted": True},
            context="test",
        )
        s = str(record)
        assert "awaiting_input" in s
        assert "interrupted" in s
        assert "test" in s

    def test_disabled_shadow_returns_none(self):
        shadow = AgentLoopShadow(enabled=False)
        result = shadow.check_divergence({"interrupted": True})
        assert result is None


# ---------------------------------------------------------------------------
# Invariant checking
# ---------------------------------------------------------------------------

class TestInvariantChecking:
    def test_no_violations(self):
        shadow = AgentLoopShadow()
        violations = shadow.check_invariants()
        assert violations == []
        assert shadow.invariant_violation_count == 0

    def test_disabled_returns_empty(self):
        shadow = AgentLoopShadow(enabled=False)
        violations = shadow.check_invariants()
        assert violations == []


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_shadow_survives_bad_transition(self):
        shadow = AgentLoopShadow()
        # Try to dispatch tools from AWAITING_INPUT (invalid)
        result = shadow.mirror_flag_change("tool_dispatch", True)
        # Should not crash, returns None
        assert result is None
        assert shadow.error_count >= 1

    def test_shadow_survives_repeated_errors(self):
        shadow = AgentLoopShadow()
        for _ in range(10):
            shadow.mirror_flag_change("tool_dispatch", True)
        # Should still be enabled (under 50 errors)
        assert shadow.enabled is True

    def test_auto_disable_on_many_errors(self):
        shadow = AgentLoopShadow()
        shadow._error_count = 49
        # This one pushes past 50
        shadow.mirror_flag_change("tool_dispatch", True)
        assert shadow.enabled is False
        assert shadow.error_count >= 50

    def test_summary_after_errors(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("tool_dispatch", True)
        s = shadow.summary()
        assert s["errors"] >= 1
        assert "enabled" in s
        assert "state" in s


# ---------------------------------------------------------------------------
# _derive_expected_state
# ---------------------------------------------------------------------------

class TestDeriveExpectedState:
    def test_interrupted(self):
        assert _derive_expected_state({"interrupted": True}) == AgentLoopState.INTERRUPTED

    def test_budget_exceeded(self):
        assert _derive_expected_state({"budget_exceeded": True}) == AgentLoopState.BUDGET_EXHAUSTED

    def test_compressed(self):
        assert _derive_expected_state({
            "restart_with_compressed_messages": True,
        }) == AgentLoopState.COMPRESSING_CONTEXT

    def test_continuation(self):
        assert _derive_expected_state({
            "restart_with_length_continuation": True,
        }) == AgentLoopState.HANDLING_CONTINUATION

    def test_executing_tools(self):
        assert _derive_expected_state({
            "executing_tools": True,
        }) == AgentLoopState.EXECUTING_TOOLS

    def test_processing_response(self):
        assert _derive_expected_state({
            "has_response": True,
            "response_processed": False,
        }) == AgentLoopState.PROCESSING_RESPONSE

    def test_error_recovery(self):
        assert _derive_expected_state({
            "in_retry": True,
        }) == AgentLoopState.ERROR_RECOVERY

    def test_api_call_in_flight(self):
        assert _derive_expected_state({
            "api_call_in_flight": True,
        }) == AgentLoopState.CALLING_API

    def test_preparing_call(self):
        assert _derive_expected_state({
            "preparing_call": True,
        }) == AgentLoopState.PREPARING_API_CALL

    def test_loop_exited(self):
        assert _derive_expected_state({
            "loop_exited": True,
        }) == AgentLoopState.RETURNING_RESPONSE

    def test_default_awaiting_input(self):
        assert _derive_expected_state({}) == AgentLoopState.AWAITING_INPUT

    def test_priority_interrupted_over_budget(self):
        """Interrupted takes priority over budget_exceeded."""
        assert _derive_expected_state({
            "interrupted": True,
            "budget_exceeded": True,
        }) == AgentLoopState.INTERRUPTED

    def test_priority_budget_over_compression(self):
        """Budget exceeded takes priority over compression."""
        assert _derive_expected_state({
            "budget_exceeded": True,
            "restart_with_compressed_messages": True,
        }) == AgentLoopState.BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# _states_compatible
# ---------------------------------------------------------------------------

class TestStatesCompatible:
    def test_exact_match(self):
        assert _states_compatible(
            AgentLoopState.AWAITING_INPUT,
            AgentLoopState.AWAITING_INPUT,
            {},
        ) is True

    def test_preparing_from_awaiting(self):
        assert _states_compatible(
            AgentLoopState.PREPARING_API_CALL,
            AgentLoopState.AWAITING_INPUT,
            {},
        ) is True

    def test_preparing_from_continuation(self):
        assert _states_compatible(
            AgentLoopState.PREPARING_API_CALL,
            AgentLoopState.HANDLING_CONTINUATION,
            {},
        ) is True

    def test_incompatible_states(self):
        assert _states_compatible(
            AgentLoopState.EXECUTING_TOOLS,
            AgentLoopState.AWAITING_INPUT,
            {},
        ) is False

    def test_calling_from_preparing(self):
        assert _states_compatible(
            AgentLoopState.CALLING_API,
            AgentLoopState.PREPARING_API_CALL,
            {},
        ) is True


# ---------------------------------------------------------------------------
# Full loop simulation
# ---------------------------------------------------------------------------

class TestFullLoopSimulation:
    def test_simple_text_response_loop(self):
        """Simulate: user message -> API call -> text response -> return."""
        shadow = AgentLoopShadow()

        # 1. Receive message
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hello"}])
        assert shadow.state == AgentLoopState.PREPARING_API_CALL.value

        # 2. API call
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": [{"role": "user", "content": "hello"}]})
        assert shadow.state == AgentLoopState.CALLING_API.value

        # 3. Response received
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        assert shadow.state == AgentLoopState.PROCESSING_RESPONSE.value

        # 4. Text response (no tools)
        shadow.mirror_flag_change("text_response", True)
        assert shadow.state == AgentLoopState.HANDLING_CONTINUATION.value

        # 5. Return result
        shadow.sm.loop_context.finish_reason = "stop"
        shadow.mirror_flag_change("return_result", True)
        assert shadow.state == AgentLoopState.RETURNING_RESPONSE.value

        # Verify no divergences
        assert shadow.divergence_count == 0
        assert shadow.transition_count == 5

    def test_tool_call_loop(self):
        """Simulate: user message -> API call -> tool call -> tool result -> API call -> text -> return."""
        shadow = AgentLoopShadow()

        # 1. Receive message
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "search for test"}])

        # 2. API call
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})

        # 3. Response with tool calls
        shadow.sm.loop_context.pending_tool_calls = [{"id": "1"}]
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop",
            tool_calls=[{"id": "1", "name": "search"}])

        # 4. Dispatch tools
        shadow.mirror_flag_change("tool_dispatch", True)
        assert shadow.state == AgentLoopState.EXECUTING_TOOLS.value

        # 5. Tools complete
        shadow.mirror_flag_change("tools_complete", True,
            results=[{"role": "tool", "tool_call_id": "1", "content": "found"}])
        assert shadow.state == AgentLoopState.HANDLING_CONTINUATION.value

        # 6. Continue (needs another API call)
        shadow.sm.loop_context.force_continuation = True
        shadow.mirror_flag_change("restart_with_length_continuation", True)
        assert shadow.state == AgentLoopState.PREPARING_API_CALL.value

        # 7. Second API call
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})

        # 8. Text response
        shadow.sm.loop_context.force_continuation = False
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        shadow.mirror_flag_change("text_response", True)

        # 9. Return
        shadow.sm.loop_context.finish_reason = "stop"
        shadow.mirror_flag_change("return_result", True)
        assert shadow.state == AgentLoopState.RETURNING_RESPONSE.value
        assert shadow.divergence_count == 0

    def test_interrupt_during_api_call(self):
        """Simulate: user message -> API call -> interrupt."""
        shadow = AgentLoopShadow()

        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})

        # Interrupt fires
        shadow.mirror_flag_change("interrupted", True)
        assert shadow.state == AgentLoopState.INTERRUPTED.value

    def test_api_error_with_retry(self):
        """Simulate: API call -> error -> retry -> success."""
        shadow = AgentLoopShadow()

        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("api_call_started", True,
            api_params={"messages": []})

        # API error
        shadow.mirror_flag_change("api_error", True,
            error=Exception("timeout"))
        assert shadow.state == AgentLoopState.ERROR_RECOVERY.value

        # Retry
        shadow.mirror_flag_change("retry_api", True)
        assert shadow.state == AgentLoopState.CALLING_API.value

        # Success
        shadow.mirror_flag_change("response_received", True,
            response={}, finish_reason="stop")
        assert shadow.state == AgentLoopState.PROCESSING_RESPONSE.value

    def test_summary_after_session(self):
        """Verify summary reflects session activity."""
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        shadow.mirror_flag_change("interrupted", True)

        s = shadow.summary()
        assert s["enabled"] is True
        assert s["transitions"] == 2
        assert s["divergences"] == 0
        assert s["errors"] == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_to_awaiting(self):
        shadow = AgentLoopShadow()
        shadow.mirror_flag_change("messages_received", True,
            messages=[{"role": "user", "content": "hi"}])
        assert shadow.state == AgentLoopState.PREPARING_API_CALL.value
        shadow.reset()
        assert shadow.state == AgentLoopState.AWAITING_INPUT.value

    def test_force_state(self):
        shadow = AgentLoopShadow()
        shadow.force_state(AgentLoopState.EXECUTING_TOOLS, "test")
        assert shadow.state == AgentLoopState.EXECUTING_TOOLS.value
