"""
Refactored agent loop using the AgentLoopStateMachine.

This module shows how the ~1700-line while loop in run_agent.py maps
to a state-machine-driven architecture.  Each state corresponds to a
handler method, and the state machine drives the loop instead of boolean
flags.

FLAG-TO-STATE MAPPING
=====================

Old boolean flags (run_agent.py)       ->  New explicit state
----------------------------------------------
(top of while, fresh entry)            ->  AWAITING_INPUT
(building api_params dict)             ->  PREPARING_API_CALL
(inside client.messages.create)        ->  CALLING_API
(parsing response object)              ->  PROCESSING_RESPONSE
(for tool_call in tool_calls:)         ->  EXECUTING_TOOLS
restart_with_compressed_messages=True   ->  COMPRESSING_CONTEXT
restart_with_length_continuation=True   ->  HANDLING_CONTINUATION
budget_exceeded=True                   ->  BUDGET_EXHAUSTED
interrupted=True                       ->  INTERRUPTED
(in except block, retry logic)         ->  ERROR_RECOVERY
(break, return result)                 ->  RETURNING_RESPONSE

CONTROL FLOW MAPPING
====================

Old code pattern:
    while True:
        if restart_with_compressed_messages:
            restart_with_compressed_messages = False
            messages = compress(messages)
            continue
        if restart_with_length_continuation:
            restart_with_length_continuation = False
            continue
        if budget_exceeded:
            break
        # ... build params, call API, parse response ...
        for tool_call in tool_calls:
            # execute tool
        if len(messages) > threshold:
            restart_with_compressed_messages = True
            continue
        if finish_reason == "length":
            restart_with_length_continuation = True
            continue
        break

New code pattern:
    sm = AgentLoopStateMachine(...)
    sm.receive_message(messages)
    while not sm.is_terminal:
        handler = STATE_HANDLERS[sm.state]
        await handler(sm, ...)
    return sm.loop_context.last_response

Usage:
    from agent_loop_refactor import run_agent_loop

    result = await run_agent_loop(
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        iteration_budget=25,
        api_client=client,
        tool_registry=tools,
    )
"""

import asyncio
import logging
import signal
from typing import Any, Callable, Coroutine, Dict, List, Optional

from agent_loop_state_machine import (
    Action,
    AgentLoopContext,
    AgentLoopState,
    AgentLoopStateMachine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# An API client callable: (params) -> response_dict
ApiClientFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]

# A tool executor: (tool_name, tool_input) -> result_dict
ToolExecutorFn = Callable[
    [str, Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]
]

# A compression function: (messages, threshold) -> compressed_messages
CompressFn = Callable[
    [List[Dict[str, Any]], int], Coroutine[Any, Any, List[Dict[str, Any]]]
]


# ---------------------------------------------------------------------------
# State handler protocol
# ---------------------------------------------------------------------------

class AgentLoopRunner:
    """Runs the agent loop using the state machine.

    Each state maps to a handler method. The main loop is just:
        while not sm.is_terminal:
            await self._handlers[sm.state](sm)

    This replaces ~1700 lines of while-loop with boolean flags.
    """

    def __init__(
        self,
        api_client: ApiClientFn,
        tool_executor: ToolExecutorFn,
        compress_fn: CompressFn,
        build_params_fn: Optional[Callable] = None,
    ):
        """
        Args:
            api_client: Async callable that sends API requests.
            tool_executor: Async callable that executes a single tool.
            compress_fn: Async callable that compresses the message list.
            build_params_fn: Optional callable to build API params from context.
        """
        self._api_client = api_client
        self._tool_executor = tool_executor
        self._compress_fn = compress_fn
        self._build_params_fn = build_params_fn

        # State -> handler mapping
        # Each handler advances the state machine by one step.
        self._handlers: Dict[
            AgentLoopState,
            Callable[[AgentLoopStateMachine], Coroutine],
        ] = {
            AgentLoopState.PREPARING_API_CALL: self._handle_preparing_api_call,
            AgentLoopState.CALLING_API: self._handle_calling_api,
            AgentLoopState.PROCESSING_RESPONSE: self._handle_processing_response,
            AgentLoopState.EXECUTING_TOOLS: self._handle_executing_tools,
            AgentLoopState.HANDLING_CONTINUATION: self._handle_continuation,
            AgentLoopState.COMPRESSING_CONTEXT: self._handle_compressing,
            AgentLoopState.ERROR_RECOVERY: self._handle_error_recovery,
            # Terminal states have no handlers (loop exits)
        }

    # ---- Main entry point ----

    async def run(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        iteration_budget: int = 25,
        max_retries: int = 3,
        max_tokens: int = 200_000,
        compression_threshold: int = 150_000,
    ) -> Dict[str, Any]:
        """Run the agent loop to completion.

        This is the top-level replacement for the old while loop.

        Args:
            messages: Initial message list (must be non-empty).
            system_prompt: Optional system prompt.
            iteration_budget: Max iterations before budget exhaustion.
            max_retries: Max API retries on error.
            max_tokens: Token budget.
            compression_threshold: Token count triggering compression.

        Returns:
            Dict with keys: response, messages, state, iterations_used
        """
        sm = AgentLoopStateMachine(
            iteration_budget=iteration_budget,
            max_retries=max_retries,
            max_tokens=max_tokens,
            compression_threshold=compression_threshold,
        )

        # Set up interrupt handler
        interrupted = False

        def _on_interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _on_interrupt)

        try:
            # Transition: AWAITING_INPUT -> PREPARING_API_CALL
            sm.receive_message(messages, system_prompt)

            # Main loop: dispatch to state handlers
            while not sm.is_terminal:
                # Check for interrupt before each step
                if interrupted:
                    sm.interrupt()
                    break

                state = sm.state
                handler = self._handlers.get(state)

                if handler is None:
                    logger.error(
                        "No handler for state %s, forcing RETURNING_RESPONSE",
                        state,
                    )
                    sm.force_state(
                        AgentLoopState.RETURNING_RESPONSE,
                        reason=f"no handler for {state}",
                    )
                    break

                await handler(sm)

            # Build result
            ctx = sm.loop_context
            return {
                "response": ctx.last_response,
                "messages": ctx.messages,
                "state": sm.state.value,
                "iterations_used": ctx.iterations_used,
                "interrupted": sm.state == AgentLoopState.INTERRUPTED,
                "budget_exhausted": sm.state == AgentLoopState.BUDGET_EXHAUSTED,
            }

        finally:
            signal.signal(signal.SIGINT, old_handler)

    # ---- State handlers ----

    async def _handle_preparing_api_call(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """PREPARING_API_CALL: Build API params and transition to CALLING_API.

        Old code location: the section that constructs api_params dict
        from messages, system prompt, model config, etc.

        If budget is exhausted, transitions to BUDGET_EXHAUSTED instead.
        """
        ctx = sm.loop_context

        if not ctx.has_budget:
            sm.exhaust_budget()
            return

        # Build API parameters
        if self._build_params_fn:
            api_params = self._build_params_fn(ctx)
        else:
            api_params = {
                "messages": ctx.messages,
                "system": ctx.system_prompt,
            }

        # Transition: PREPARING_API_CALL -> CALLING_API
        sm.build_request(api_params)

    async def _handle_calling_api(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """CALLING_API: Send the API request and receive response.

        Old code location: `response = await client.messages.create(**params)`

        On success -> PROCESSING_RESPONSE
        On error   -> ERROR_RECOVERY
        """
        ctx = sm.loop_context

        try:
            response = await self._api_client(ctx.api_params)

            # Extract standard fields from response
            finish_reason = response.get("finish_reason", "stop")
            tool_calls = response.get("tool_calls", [])
            token_count = response.get("usage", {}).get(
                "total_tokens", ctx.token_count
            )

            # Transition: CALLING_API -> PROCESSING_RESPONSE
            sm.receive_response(
                response=response,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                token_count=token_count,
            )

        except Exception as e:
            logger.warning("API call failed: %s", e)
            # Transition: CALLING_API -> ERROR_RECOVERY
            sm.recover_error(e)

    async def _handle_processing_response(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """PROCESSING_RESPONSE: Decide whether to execute tools or return.

        Old code location: the if/elif block after response parsing that
        checks `if tool_calls:` vs text-only response.

        With tools    -> EXECUTING_TOOLS
        Without tools -> HANDLING_CONTINUATION
        """
        action = sm.decide_after_response()

        if action == Action.DISPATCH_TOOLS:
            # Append assistant message with tool calls to messages
            ctx = sm.loop_context
            ctx.messages.append({
                "role": "assistant",
                "content": ctx.last_response.get("content", ""),
                "tool_calls": ctx.pending_tool_calls,
            })
            sm.dispatch_tools()
        else:
            # Text-only response
            ctx = sm.loop_context
            ctx.messages.append({
                "role": "assistant",
                "content": ctx.last_response.get("content", ""),
            })
            sm.process_text_response()

    async def _handle_executing_tools(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """EXECUTING_TOOLS: Run each tool call and collect results.

        Old code location: `for tool_call in tool_calls:` loop that
        dispatches to tool handlers and collects results.

        Always -> HANDLING_CONTINUATION (via tool_complete)
        """
        ctx = sm.loop_context
        results = []

        for tool_call in ctx.pending_tool_calls:
            tool_name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
            tool_input = tool_call.get("input", tool_call.get("function", {}).get("arguments", {}))
            tool_id = tool_call.get("id", "")

            try:
                result = await self._tool_executor(tool_name, tool_input)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result.get("content", str(result)),
                })
            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: {e}",
                    "is_error": True,
                })

        # Transition: EXECUTING_TOOLS -> HANDLING_CONTINUATION
        sm.tool_complete(results)

    async def _handle_continuation(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """HANDLING_CONTINUATION: Decide next step after tools or text.

        This replaces the old pattern:
            if budget_exceeded:
                break
            if restart_with_compressed_messages:
                restart_with_compressed_messages = False
                messages = compress(messages)
                continue
            if restart_with_length_continuation:
                restart_with_length_continuation = False
                continue
            break  # done

        Budget exhausted      -> BUDGET_EXHAUSTED
        Context too large     -> COMPRESSING_CONTEXT
        Needs continuation    -> PREPARING_API_CALL
        Turn done             -> RETURNING_RESPONSE
        """
        action = sm.decide_after_continuation()

        if action == Action.EXHAUST_BUDGET:
            sm.exhaust_budget()
        elif action == Action.COMPRESS:
            sm.compress(sm.loop_context.messages, sm.loop_context.token_count)
            # Compression happens in next handler
        elif action == Action.CONTINUE_GENERATION:
            # Force continuation flag for the next iteration
            sm.continue_generation()
        elif action == Action.RETURN_RESULT:
            sm.return_result()

    async def _handle_compressing(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """COMPRESSING_CONTEXT: Compress messages and return to preparing.

        Old code: restart_with_compressed_messages = True; messages = compress(messages)

        Always -> PREPARING_API_CALL (via continue_after_compression)
        """
        ctx = sm.loop_context

        compressed = await self._compress_fn(
            ctx.messages, ctx.compression_threshold
        )

        # Update context with compressed messages
        # Estimate new token count (proportional reduction)
        if ctx.messages:
            ratio = len(compressed) / max(len(ctx.messages), 1)
        else:
            ratio = 1.0
        new_token_count = int(ctx.token_count * ratio)

        ctx.messages = compressed
        ctx.token_count = new_token_count

        # Transition: COMPRESSING_CONTEXT -> PREPARING_API_CALL
        sm.continue_after_compression()

    async def _handle_error_recovery(
        self, sm: AgentLoopStateMachine
    ) -> None:
        """ERROR_RECOVERY: Retry or give up.

        Old code: the except block that checks retry_count < max_retries.

        Can retry     -> CALLING_API (via retry_api)
        Retries spent -> RETURNING_RESPONSE (via return_result)
        """
        action = sm.decide_after_error()

        if action == Action.RETRY_API:
            ctx = sm.loop_context
            # Exponential backoff
            backoff = min(2 ** ctx.retry_count, 30)
            logger.info(
                "Retrying API call in %ds (attempt %d/%d)",
                backoff,
                ctx.retry_count + 1,
                ctx.max_retries,
            )
            await asyncio.sleep(backoff)
            sm.retry_api()
        else:
            logger.error(
                "API retries exhausted (%d/%d), returning error",
                sm.loop_context.retry_count,
                sm.loop_context.max_retries,
            )
            sm.return_result()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def run_agent_loop(
    messages: List[Dict[str, Any]],
    api_client: ApiClientFn,
    tool_executor: ToolExecutorFn,
    compress_fn: CompressFn,
    system_prompt: Optional[str] = None,
    iteration_budget: int = 25,
    max_retries: int = 3,
    max_tokens: int = 200_000,
    compression_threshold: int = 150_000,
    build_params_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run the agent loop (convenience wrapper).

    Args:
        messages: Initial messages (non-empty).
        api_client: Async API client callable.
        tool_executor: Async tool executor callable.
        compress_fn: Async message compression callable.
        system_prompt: Optional system prompt.
        iteration_budget: Max iterations.
        max_retries: Max API retries.
        max_tokens: Token budget.
        compression_threshold: When to compress.
        build_params_fn: Optional API param builder.

    Returns:
        Result dict with response, messages, state, etc.
    """
    runner = AgentLoopRunner(
        api_client=api_client,
        tool_executor=tool_executor,
        compress_fn=compress_fn,
        build_params_fn=build_params_fn,
    )
    return await runner.run(
        messages=messages,
        system_prompt=system_prompt,
        iteration_budget=iteration_budget,
        max_retries=max_retries,
        max_tokens=max_tokens,
        compression_threshold=compression_threshold,
    )
