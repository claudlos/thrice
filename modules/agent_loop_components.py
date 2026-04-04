"""
Agent Loop Components for Hermes (Improvement #6: Agent Loop Decomposition).

Extracts clean, testable components from the monolithic run_agent.py into
standalone classes that COULD replace inline logic. These are NOT a refactor
of run_agent.py itself — they are independent, composable building blocks.

Components:
    ToolDispatcher   — Tool name repair, argument validation, dispatch
    RetryEngine      — Retry logic with backoff strategies and error classification
    MessageProcessor — Message validation, trimming, injection, extraction
    IterationTracker — Loop iteration tracking, stuck detection, escape suggestions
    CostTracker      — Token cost tracking with per-model pricing

Usage:
    from agent_loop_components import (
        ToolDispatcher, ToolResult,
        RetryEngine, RetryConfig,
        MessageProcessor,
        IterationTracker,
        CostTracker,
    )
"""

import enum
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a tool dispatch."""
    success: bool
    result: str
    tool_name: str
    duration: float
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


# Common tool name aliases (old name -> canonical name)
_DEFAULT_TOOL_ALIASES: Dict[str, str] = {
    "bash": "terminal",
    "shell": "terminal",
    "exec": "terminal",
    "run": "terminal",
    "sh": "terminal",
    "edit": "str_replace_editor",
    "replace": "str_replace_editor",
    "sed": "str_replace_editor",
    "cat": "read_file",
    "view": "read_file",
    "show": "read_file",
    "grep": "search",
    "find": "search",
    "rg": "search",
    "write": "write_file",
    "create": "write_file",
    "save": "write_file",
}


class ToolDispatcher:
    """Dispatches tool calls with name repair and argument validation.

    Handles the common failure modes in agent loops:
    - Model hallucinates a tool name that doesn't exist
    - Model provides wrong argument names or types
    - Tool execution raises an exception
    """

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        similarity_threshold: float = 0.6,
    ):
        self._aliases = dict(_DEFAULT_TOOL_ALIASES)
        if aliases:
            self._aliases.update(aliases)
        self._similarity_threshold = similarity_threshold

    def dispatch(
        self,
        tool_name: str,
        args: dict,
        available_tools: dict,
    ) -> ToolResult:
        """Dispatch a tool call, repairing the name if needed.

        Args:
            tool_name: The tool name from the model's response.
            args: Arguments dict for the tool.
            available_tools: Mapping of tool_name -> callable.

        Returns:
            ToolResult with success/failure and timing info.
        """
        start = time.monotonic()

        # Try to repair the tool name if it's not in available_tools
        resolved_name = tool_name
        if tool_name not in available_tools:
            resolved_name = self.repair_tool_name(
                tool_name, list(available_tools.keys())
            )
            if resolved_name not in available_tools:
                return ToolResult(
                    success=False,
                    result="",
                    tool_name=tool_name,
                    duration=time.monotonic() - start,
                    error=f"Unknown tool: {tool_name!r} (tried repair to {resolved_name!r})",
                )
            logger.info("Repaired tool name %r -> %r", tool_name, resolved_name)

        tool_fn = available_tools[resolved_name]

        try:
            result = tool_fn(**args)
            duration = time.monotonic() - start
            return ToolResult(
                success=True,
                result=str(result) if result is not None else "",
                tool_name=resolved_name,
                duration=duration,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return ToolResult(
                success=False,
                result="",
                tool_name=resolved_name,
                duration=duration,
                error=f"{type(exc).__name__}: {exc}",
            )

    def repair_tool_name(
        self,
        name: str,
        available_tools: List[str],
    ) -> str:
        """Attempt to repair a tool name using aliases and fuzzy matching.

        Strategy order:
        1. Exact match (identity)
        2. Case-insensitive match
        3. Alias lookup
        4. Fuzzy match (SequenceMatcher)

        Returns the best candidate or the original name if nothing matches.
        """
        if not available_tools:
            return name

        # 1. Exact match
        if name in available_tools:
            return name

        # 2. Case-insensitive
        lower_map = {t.lower(): t for t in available_tools}
        if name.lower() in lower_map:
            return lower_map[name.lower()]

        # 3. Alias lookup
        alias_target = self._aliases.get(name.lower())
        if alias_target and alias_target in available_tools:
            return alias_target
        # Also check if alias target matches case-insensitively
        if alias_target and alias_target.lower() in lower_map:
            return lower_map[alias_target.lower()]

        # 4. Fuzzy match
        best_score = 0.0
        best_match = name
        for candidate in available_tools:
            score = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= self._similarity_threshold:
            return best_match

        return name

    def validate_args(
        self,
        tool_name: str,
        args: dict,
        schema: dict,
    ) -> List[str]:
        """Validate tool arguments against a schema.

        Schema format (simplified JSON Schema subset):
            {
                "required": ["param1", "param2"],
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"},
                    "param3": {"type": "boolean", "default": False},
                }
            }

        Returns a list of validation error strings (empty = valid).
        """
        errors: List[str] = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required params
        for param in required:
            if param not in args:
                errors.append(f"Missing required parameter: {param!r}")

        # Check types
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for param_name, value in args.items():
            if param_name not in properties:
                errors.append(f"Unknown parameter: {param_name!r}")
                continue

            expected_type_str = properties[param_name].get("type")
            if expected_type_str and expected_type_str in type_map:
                expected = type_map[expected_type_str]
                if not isinstance(value, expected):
                    errors.append(
                        f"Parameter {param_name!r}: expected {expected_type_str}, "
                        f"got {type(value).__name__}"
                    )

        return errors


# ---------------------------------------------------------------------------
# RetryEngine
# ---------------------------------------------------------------------------

class ErrorClass(enum.Enum):
    """Classification of errors for retry decisions."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: str = "exponential"  # exponential, linear, constant
    retryable_errors: Optional[Set[str]] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = {
                ErrorClass.TRANSIENT.value,
                ErrorClass.RATE_LIMIT.value,
            }
        if self.strategy not in ("exponential", "linear", "constant"):
            raise ValueError(f"Unknown retry strategy: {self.strategy!r}")


# Keywords in error messages that hint at error classification
_RATE_LIMIT_KEYWORDS = {"rate limit", "rate_limit", "429", "too many requests", "quota"}
_AUTH_KEYWORDS = {"auth", "401", "403", "forbidden", "unauthorized", "api key", "permission"}
_CONTEXT_KEYWORDS = {"context", "token limit", "too long", "max_tokens", "context_length", "context window"}
_TRANSIENT_KEYWORDS = {"timeout", "connection", "502", "503", "504", "temporary", "unavailable", "overloaded"}


class RetryEngine:
    """Retry logic with backoff strategies and error classification.

    Replaces the ad-hoc retry logic scattered through run_agent.py's
    exception handlers with a single, testable engine.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self._config = config or RetryConfig()

    @property
    def config(self) -> RetryConfig:
        return self._config

    def classify_error(self, error: Exception) -> str:
        """Classify an error into a category for retry decisions.

        Returns one of: transient, permanent, rate_limit, auth, context_overflow
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        combined = f"{error_type} {error_str}"

        # Check for specific HTTP status codes in the error
        if any(kw in combined for kw in _RATE_LIMIT_KEYWORDS):
            return ErrorClass.RATE_LIMIT.value

        if any(kw in combined for kw in _AUTH_KEYWORDS):
            return ErrorClass.AUTH.value

        if any(kw in combined for kw in _CONTEXT_KEYWORDS):
            return ErrorClass.CONTEXT_OVERFLOW.value

        if any(kw in combined for kw in _TRANSIENT_KEYWORDS):
            return ErrorClass.TRANSIENT.value

        # Network-level errors are generally transient
        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return ErrorClass.TRANSIENT.value

        return ErrorClass.PERMANENT.value

    def should_retry(
        self,
        error: Exception,
        attempt: int,
        max_retries: Optional[int] = None,
    ) -> bool:
        """Determine whether to retry after an error.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (0-based).
            max_retries: Override for config max_retries.
        """
        max_r = max_retries if max_retries is not None else self._config.max_retries
        if attempt >= max_r:
            return False

        classification = self.classify_error(error)
        return classification in self._config.retryable_errors

    def get_delay(
        self,
        attempt: int,
        strategy: Optional[str] = None,
    ) -> float:
        """Calculate delay before the next retry attempt.

        Args:
            attempt: Current attempt number (0-based).
            strategy: Override for config strategy.

        Returns:
            Delay in seconds, capped at max_delay.
        """
        strat = strategy or self._config.strategy
        base = self._config.base_delay
        max_d = self._config.max_delay

        if strat == "constant":
            delay = base
        elif strat == "linear":
            delay = base * (attempt + 1)
        elif strat == "exponential":
            delay = base * (2 ** attempt)
        else:
            raise ValueError(f"Unknown strategy: {strat!r}")

        return min(delay, max_d)

    def execute_with_retry(
        self,
        fn: Callable,
        config: Optional[RetryConfig] = None,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            fn: Callable to execute (no arguments).
            config: Override retry config.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        cfg = config or self._config
        last_error: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                classification = self.classify_error(exc)
                logger.warning(
                    "Attempt %d/%d failed (%s: %s), class=%s",
                    attempt + 1, cfg.max_retries + 1,
                    type(exc).__name__, exc, classification,
                )

                if attempt >= cfg.max_retries:
                    break
                if classification not in cfg.retryable_errors:
                    break

                delay = self.get_delay(attempt, cfg.strategy)
                delay = min(delay, cfg.max_delay)
                logger.info("Retrying in %.1fs...", delay)
                time.sleep(delay)

        raise last_error


# ---------------------------------------------------------------------------
# MessageProcessor
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant", "tool"}


class MessageProcessor:
    """Validates, trims, and manipulates message lists.

    Handles the common message-list operations that are scattered
    through run_agent.py: validation, budget trimming, system note
    injection, and tool call extraction.
    """

    def validate_messages(self, messages: List[dict]) -> List[str]:
        """Validate a message list for common issues.

        Returns a list of error strings (empty = valid).
        """
        errors: List[str] = []

        if not messages:
            errors.append("Message list is empty")
            return errors

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}]: not a dict")
                continue

            if "role" not in msg:
                errors.append(f"messages[{i}]: missing 'role'")
                continue

            role = msg["role"]
            if role not in _VALID_ROLES:
                errors.append(f"messages[{i}]: invalid role {role!r}")

            if "content" not in msg and "tool_calls" not in msg:
                # tool role messages need tool_call_id
                if role == "tool":
                    if "tool_call_id" not in msg:
                        errors.append(
                            f"messages[{i}]: tool message missing 'tool_call_id'"
                        )
                elif role != "assistant":
                    errors.append(
                        f"messages[{i}]: missing 'content' (role={role!r})"
                    )

        # Check system message is first if present
        system_indices = [
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "system"
        ]
        if system_indices and system_indices[0] != 0:
            errors.append("System message must be first in the list")

        return errors

    def trim_to_budget(
        self,
        messages: List[dict],
        max_tokens: int,
        token_counter: Callable[[List[dict]], int],
    ) -> List[dict]:
        """Trim messages to fit within a token budget.

        Strategy: preserve system message (first) and recent messages,
        remove oldest non-system messages first.

        Args:
            messages: Full message list.
            max_tokens: Maximum token budget.
            token_counter: Function that counts tokens for a message list.

        Returns:
            Trimmed message list.
        """
        if not messages:
            return messages

        current_count = token_counter(messages)
        if current_count <= max_tokens:
            return list(messages)

        # Preserve system message if first
        preserved_start: List[dict] = []
        trimmable: List[dict] = []

        if messages[0].get("role") == "system":
            preserved_start = [messages[0]]
            trimmable = list(messages[1:])
        else:
            trimmable = list(messages)

        # Remove oldest messages until within budget
        while trimmable and token_counter(preserved_start + trimmable) > max_tokens:
            trimmable.pop(0)

        return preserved_start + trimmable

    def inject_system_note(
        self,
        messages: List[dict],
        note: str,
    ) -> List[dict]:
        """Inject a system-level note into the message list.

        If a system message exists, appends to its content.
        Otherwise, prepends a new system message.

        Returns a new list (does not mutate input).
        """
        result = list(messages)

        if result and result[0].get("role") == "system":
            existing = result[0].get("content", "")
            separator = "\n\n" if existing else ""
            result[0] = {**result[0], "content": f"{existing}{separator}{note}"}
        else:
            result.insert(0, {"role": "system", "content": note})

        return result

    def extract_tool_calls(self, message: dict) -> List[dict]:
        """Extract tool calls from an assistant message.

        Handles both OpenAI-style and Anthropic-style formats.

        Returns list of dicts with keys: id, name, arguments.
        """
        calls: List[dict] = []

        # OpenAI format: message.tool_calls
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                call = {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                }
                # Parse arguments if they're a string
                if isinstance(call["arguments"], str):
                    try:
                        call["arguments"] = json.loads(call["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                calls.append(call)

        # Anthropic format: content blocks with type=tool_use
        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    })

        return calls

    def format_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> dict:
        """Format a tool result as a message dict (OpenAI format).

        Returns:
            Dict suitable for appending to the messages list.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }


# ---------------------------------------------------------------------------
# IterationTracker
# ---------------------------------------------------------------------------

@dataclass
class _ToolCallRecord:
    """Internal record of a tool call."""
    tool_name: str
    args_hash: str
    result_hash: str
    duration: float
    timestamp: float


@dataclass
class _ApiCallRecord:
    """Internal record of an API call."""
    model: str
    input_tokens: int
    output_tokens: int
    duration: float
    timestamp: float


@dataclass
class _IterationRecord:
    """Internal record of a single iteration."""
    iteration_num: int
    start_time: float
    tool_calls: List[_ToolCallRecord] = field(default_factory=list)
    api_calls: List[_ApiCallRecord] = field(default_factory=list)


def _hash_args(args: dict) -> str:
    """Create a stable hash of tool arguments for similarity detection."""
    try:
        serialized = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(args)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _hash_result(result: str) -> str:
    """Create a stable hash of a tool result."""
    return hashlib.sha256(result.encode()).hexdigest()


class IterationTracker:
    """Tracks agent loop iterations, detecting stuck loops and providing stats.

    Replaces the manual iteration counting and ad-hoc stuck detection
    in run_agent.py with structured tracking and analysis.
    """

    def __init__(self):
        self._iterations: List[_IterationRecord] = []
        self._current: Optional[_IterationRecord] = None
        self._start_time: float = time.monotonic()

    def start_iteration(self, iteration_num: int) -> None:
        """Mark the start of a new iteration."""
        self._current = _IterationRecord(
            iteration_num=iteration_num,
            start_time=time.monotonic(),
        )
        self._iterations.append(self._current)

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        duration: float,
    ) -> None:
        """Record a tool call in the current iteration."""
        if self._current is None:
            self.start_iteration(len(self._iterations))

        record = _ToolCallRecord(
            tool_name=tool_name,
            args_hash=_hash_args(args),
            result_hash=_hash_result(result),
            duration=duration,
            timestamp=time.monotonic(),
        )
        self._current.tool_calls.append(record)

    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
    ) -> None:
        """Record an API call in the current iteration."""
        if self._current is None:
            self.start_iteration(len(self._iterations))

        record = _ApiCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration=duration,
            timestamp=time.monotonic(),
        )
        self._current.api_calls.append(record)

    def is_stuck(self, window: int = 5) -> bool:
        """Detect if the agent is stuck in a loop.

        Checks if the same tool has been called with similar arguments
        repeatedly in the last `window` iterations.

        Returns True if stuck (same tool+args pattern repeats >= window times).
        """
        if len(self._iterations) < window:
            return False

        recent = self._iterations[-window:]

        # Collect all tool call signatures from recent iterations
        signatures: List[Tuple[str, str]] = []
        for iteration in recent:
            for tc in iteration.tool_calls:
                signatures.append((tc.tool_name, tc.args_hash))

        if not signatures:
            return False

        # Check if any single signature dominates
        from collections import Counter
        counts = Counter(signatures)
        most_common_sig, most_common_count = counts.most_common(1)[0]

        # Stuck if the same tool+args appears in >= window iterations
        # (allowing for some variation)
        iterations_with_sig = 0
        for iteration in recent:
            for tc in iteration.tool_calls:
                if (tc.tool_name, tc.args_hash) == most_common_sig:
                    iterations_with_sig += 1
                    break

        return iterations_with_sig >= window

    def suggest_escape(self) -> str:
        """When stuck, suggest a different approach.

        Analyzes the recent tool call pattern and suggests alternatives.
        """
        if not self._iterations:
            return "No iterations recorded yet."

        recent = self._iterations[-5:] if len(self._iterations) >= 5 else self._iterations
        tool_names = []
        for iteration in recent:
            for tc in iteration.tool_calls:
                tool_names.append(tc.tool_name)

        if not tool_names:
            return "No tool calls recorded. Try using a tool to make progress."

        from collections import Counter
        counts = Counter(tool_names)
        most_used, count = counts.most_common(1)[0]

        suggestions = {
            "terminal": (
                "You've been running terminal commands repeatedly. "
                "Try reading the error output carefully, or use a different "
                "approach like editing the file directly."
            ),
            "str_replace_editor": (
                "You've been editing repeatedly. Step back and re-read the "
                "file to check your understanding, or try a different fix strategy."
            ),
            "read_file": (
                "You've been reading files repeatedly. Try making a change "
                "based on what you've read, or search for different files."
            ),
            "search": (
                "You've been searching repeatedly. Try narrowing your search "
                "terms, or read a specific file you've already found."
            ),
        }

        specific = suggestions.get(most_used, "")
        general = (
            f"The tool '{most_used}' has been called {count} times in "
            f"the last {len(recent)} iterations. "
            f"Consider a fundamentally different approach."
        )

        return f"{general} {specific}".strip()

    def get_stats(self) -> dict:
        """Get aggregate statistics for the session."""
        total_tool_calls = sum(
            len(it.tool_calls) for it in self._iterations
        )
        total_api_calls = sum(
            len(it.api_calls) for it in self._iterations
        )
        total_input_tokens = sum(
            ac.input_tokens
            for it in self._iterations
            for ac in it.api_calls
        )
        total_output_tokens = sum(
            ac.output_tokens
            for it in self._iterations
            for ac in it.api_calls
        )
        total_tool_duration = sum(
            tc.duration
            for it in self._iterations
            for tc in it.tool_calls
        )
        total_api_duration = sum(
            ac.duration
            for it in self._iterations
            for ac in it.api_calls
        )
        elapsed = time.monotonic() - self._start_time

        return {
            "total_iterations": len(self._iterations),
            "total_tool_calls": total_tool_calls,
            "total_api_calls": total_api_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_tool_duration": round(total_tool_duration, 3),
            "total_api_duration": round(total_api_duration, 3),
            "elapsed_seconds": round(elapsed, 3),
        }


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

# Pricing per 1M tokens (input, output) in USD
# Updated pricing as of 2025
_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Claude models
    "claude-opus-4": (15.00, 75.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-3.5": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-opus-20240229": (15.00, 75.00),
    # OpenAI models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Google models
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}


@dataclass
class _CostRecord:
    """Internal record of a single API cost entry."""
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    timestamp: float


class CostTracker:
    """Tracks token usage and costs across models.

    Provides real-time cost estimation using per-model pricing tables.
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self._pricing = dict(_MODEL_PRICING)
        if pricing:
            self._pricing.update(pricing)
        self._records: List[_CostRecord] = []

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage and return the cost for this call.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD for this API call.
        """
        input_price, output_price = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        self._records.append(_CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            timestamp=time.monotonic(),
        ))

        return input_cost + output_cost

    def get_session_cost(self) -> float:
        """Get total session cost in USD."""
        return sum(r.input_cost + r.output_cost for r in self._records)

    def get_session_stats(self) -> dict:
        """Get detailed session statistics."""
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_cost = self.get_session_cost()

        # Per-model breakdown
        model_stats: Dict[str, dict] = {}
        for r in self._records:
            if r.model not in model_stats:
                model_stats[r.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            ms = model_stats[r.model]
            ms["calls"] += 1
            ms["input_tokens"] += r.input_tokens
            ms["output_tokens"] += r.output_tokens
            ms["cost"] += r.input_cost + r.output_cost

        return {
            "total_calls": len(self._records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "models": model_stats,
        }

    def format_summary(self) -> str:
        """Format a human-readable cost summary."""
        stats = self.get_session_stats()

        if not self._records:
            return "No API calls recorded."

        lines = [
            f"Session Cost: ${stats['total_cost_usd']:.4f}",
            f"Total Calls: {stats['total_calls']}",
            f"Total Tokens: {stats['total_tokens']:,} "
            f"({stats['total_input_tokens']:,} in / "
            f"{stats['total_output_tokens']:,} out)",
        ]

        if len(stats["models"]) > 1:
            lines.append("Per-model breakdown:")
            for model, ms in sorted(stats["models"].items()):
                lines.append(
                    f"  {model}: {ms['calls']} calls, "
                    f"{ms['input_tokens'] + ms['output_tokens']:,} tokens, "
                    f"${ms['cost']:.4f}"
                )

        return "\n".join(lines)

    def _get_pricing(self, model: str) -> Tuple[float, float]:
        """Look up pricing for a model, with fuzzy fallback."""
        if model in self._pricing:
            return self._pricing[model]

        # Try prefix matching (e.g., "claude-sonnet-4-20250514" -> "claude-sonnet-4")
        for known_model in self._pricing:
            if model.startswith(known_model) or known_model.startswith(model):
                return self._pricing[known_model]

        # Default pricing (conservative estimate)
        logger.warning("Unknown model %r, using default pricing", model)
        return (10.00, 30.00)
