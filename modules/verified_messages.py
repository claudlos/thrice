"""
Verified Message Protocol (#14)

Builder-pattern message construction with type-level guarantees,
validation, and auto-fixing for LLM conversation message sequences.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class InvalidMessageSequence(Exception):
    """Raised when a message sequence violates protocol rules."""

    def __init__(self, message: str, position: int = -1, suggestion: str = ""):
        self.position = position
        self.suggestion = suggestion
        full_msg = message
        if position >= 0:
            full_msg = f"[position {position}] {message}"
        if suggestion:
            full_msg += f" Suggestion: {suggestion}"
        super().__init__(full_msg)


@dataclass
class ValidationError:
    """A single validation error in a message sequence."""
    position: int
    message: str
    severity: str  # "error" or "warning"
    suggestion: str = ""

    def __str__(self):
        prefix = f"[{self.severity.upper()} at {self.position}]"
        result = f"{prefix} {self.message}"
        if self.suggestion:
            result += f" Fix: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Result of validating a message sequence."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def __bool__(self):
        return self.valid

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return "Message sequence is valid."
        parts = []
        if not self.valid:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return "Validation: " + ", ".join(parts)


class MessageBuilder:
    """Immutable builder for constructing valid message sequences.

    Each method returns a new MessageBuilder instance. Invalid sequences
    raise InvalidMessageSequence at call time (not at build time).
    """

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        self._messages: List[Dict[str, Any]] = list(messages) if messages else []

    def _last_role(self) -> Optional[str]:
        if not self._messages:
            return None
        return self._messages[-1].get("role")

    def _last_message(self) -> Optional[Dict[str, Any]]:
        if not self._messages:
            return None
        return self._messages[-1]

    def system(self, content: str) -> "MessageBuilder":
        """Add a system message. Only valid as the first message."""
        if self._messages:
            raise InvalidMessageSequence(
                "System message can only be the first message.",
                position=len(self._messages),
                suggestion="Move system message to the beginning."
            )
        new_msgs = self._messages + [{"role": "system", "content": content}]
        return MessageBuilder(new_msgs)

    def user(self, content: str) -> "MessageBuilder":
        """Add a user message. Valid as first message or after assistant/system/tool."""
        last = self._last_role()
        if last == "user":
            raise InvalidMessageSequence(
                "Cannot add user message after another user message.",
                position=len(self._messages),
                suggestion="Add an assistant message between user messages."
            )
        new_msgs = self._messages + [{"role": "user", "content": content}]
        return MessageBuilder(new_msgs)

    def assistant(self, content: str,
                  tool_calls: Optional[List[Dict[str, Any]]] = None) -> "MessageBuilder":
        """Add an assistant message. Valid after user or tool messages."""
        last = self._last_role()
        if last is None:
            raise InvalidMessageSequence(
                "Assistant message cannot be the first message.",
                position=0,
                suggestion="Start with a system or user message."
            )
        if last == "assistant":
            raise InvalidMessageSequence(
                "Cannot add assistant message after another assistant message.",
                position=len(self._messages),
                suggestion="Add a user or tool message between assistant messages."
            )
        if last == "system":
            raise InvalidMessageSequence(
                "Assistant message should follow a user or tool message, not system.",
                position=len(self._messages),
                suggestion="Add a user message after system before assistant."
            )
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        new_msgs = self._messages + [msg]
        return MessageBuilder(new_msgs)

    def tool_result(self, tool_call_id: str, content: str) -> "MessageBuilder":
        """Add a tool result message. Only valid after assistant with matching tool_call."""
        last = self._last_message()
        if last is None:
            raise InvalidMessageSequence(
                "Tool result cannot be the first message.",
                position=0,
                suggestion="Start with system/user, then assistant with tool_calls."
            )

        # Find the most recent assistant message with tool_calls
        last_assistant_idx = None
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            raise InvalidMessageSequence(
                "Tool result requires a preceding assistant message with tool_calls.",
                position=len(self._messages),
                suggestion="Add an assistant message with tool_calls first."
            )

        assistant_msg = self._messages[last_assistant_idx]
        if "tool_calls" not in assistant_msg or not assistant_msg["tool_calls"]:
            raise InvalidMessageSequence(
                "Preceding assistant message has no tool_calls.",
                position=len(self._messages),
                suggestion="The assistant message must include tool_calls."
            )

        # Verify the tool_call_id matches one of the pending tool calls
        valid_ids = {tc.get("id") for tc in assistant_msg["tool_calls"]}
        if tool_call_id not in valid_ids:
            raise InvalidMessageSequence(
                f"tool_call_id '{tool_call_id}' not found in assistant's tool_calls. "
                f"Valid IDs: {valid_ids}",
                position=len(self._messages),
                suggestion=f"Use one of: {', '.join(str(v) for v in valid_ids)}"
            )

        # Check it hasn't already been answered
        answered_ids = set()
        for i in range(last_assistant_idx + 1, len(self._messages)):
            if self._messages[i]["role"] == "tool":
                answered_ids.add(self._messages[i].get("tool_call_id"))
        if tool_call_id in answered_ids:
            raise InvalidMessageSequence(
                f"tool_call_id '{tool_call_id}' already has a result.",
                position=len(self._messages),
                suggestion="Each tool_call should have exactly one result."
            )

        # Tool results must follow assistant or other tool results
        last_role = self._last_role()
        if last_role not in ("assistant", "tool"):
            raise InvalidMessageSequence(
                f"Tool result must follow assistant or tool message, got '{last_role}'.",
                position=len(self._messages),
                suggestion="Place tool results immediately after the assistant message."
            )

        new_msgs = self._messages + [{
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }]
        return MessageBuilder(new_msgs)

    def build(self) -> List[Dict[str, Any]]:
        """Return the validated message list."""
        if not self._messages:
            raise InvalidMessageSequence("Cannot build empty message sequence.", position=0)
        # Return a deep copy to prevent mutation
        return deepcopy(self._messages)

    def __len__(self):
        return len(self._messages)


class MessageValidator:
    """Validates existing message sequences against protocol rules."""

    def validate(self, messages: List[Dict[str, Any]]) -> ValidationResult:
        """Validate a message sequence, returning structured errors."""
        errors = []
        warnings = []

        if not messages:
            errors.append(ValidationError(
                position=0,
                message="Message sequence is empty.",
                severity="error",
                suggestion="Add at least one message."
            ))
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        # Track tool_calls that need results
        pending_tool_calls: Dict[str, int] = {}  # id -> assistant position

        for i, msg in enumerate(messages):
            role = msg.get("role")

            # Check role is valid
            if role not in ("system", "user", "assistant", "tool"):
                errors.append(ValidationError(
                    position=i,
                    message=f"Invalid role: '{role}'.",
                    severity="error",
                    suggestion="Use 'system', 'user', 'assistant', or 'tool'."
                ))
                continue

            # System must be first
            if role == "system" and i > 0:
                errors.append(ValidationError(
                    position=i,
                    message="System message must be the first message.",
                    severity="error",
                    suggestion="Move system message to position 0."
                ))

            # Check role alternation
            if i > 0:
                prev_role = messages[i - 1].get("role")

                if role == "user" and prev_role == "user":
                    errors.append(ValidationError(
                        position=i,
                        message="Consecutive user messages.",
                        severity="error",
                        suggestion="Insert an assistant message between user messages."
                    ))

                if role == "assistant" and prev_role == "assistant":
                    errors.append(ValidationError(
                        position=i,
                        message="Consecutive assistant messages.",
                        severity="error",
                        suggestion="Insert a user or tool message between assistant messages."
                    ))

            # Track tool_calls
            if role == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        pending_tool_calls[tc_id] = i

            # Validate tool results
            if role == "tool":
                tc_id = msg.get("tool_call_id")
                if not tc_id:
                    errors.append(ValidationError(
                        position=i,
                        message="Tool result missing tool_call_id.",
                        severity="error",
                        suggestion="Add tool_call_id matching an assistant's tool_call."
                    ))
                elif tc_id not in pending_tool_calls:
                    errors.append(ValidationError(
                        position=i,
                        message=f"Orphaned tool result: no matching tool_call for '{tc_id}'.",
                        severity="error",
                        suggestion="Remove this tool result or add matching tool_call to assistant."
                    ))
                else:
                    # Mark as fulfilled
                    del pending_tool_calls[tc_id]

                # Tool must follow assistant or tool
                if i > 0 and messages[i - 1].get("role") not in ("assistant", "tool"):
                    errors.append(ValidationError(
                        position=i,
                        message="Tool result must follow assistant or another tool result.",
                        severity="error",
                        suggestion="Move tool result to after the assistant message."
                    ))

        # Check for unfulfilled tool_calls
        for tc_id, pos in pending_tool_calls.items():
            warnings.append(ValidationError(
                position=pos,
                message=f"Tool call '{tc_id}' has no corresponding result.",
                severity="warning",
                suggestion=f"Add a tool result with tool_call_id='{tc_id}'."
            ))

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)


class AutoFixer:
    """Attempts to repair invalid message sequences."""

    def __init__(self):
        self.validator = MessageValidator()

    def fix(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt to repair an invalid message sequence.

        Strategies applied in order:
        1. Remove orphaned tool results
        2. Insert missing tool results (with placeholder content)
        3. Fix role alternation (insert placeholder messages)
        4. Move system messages to front
        """
        if not messages:
            return messages

        fixed = deepcopy(messages)

        # Strategy 1: Move system messages to front
        fixed = self._fix_system_position(fixed)

        # Strategy 2: Remove orphaned tool results
        fixed = self._remove_orphaned_tool_results(fixed)

        # Strategy 3: Fix role alternation
        fixed = self._fix_role_alternation(fixed)

        # Strategy 4: Insert missing tool results
        fixed = self._insert_missing_tool_results(fixed)

        return fixed

    def _fix_system_position(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Move system messages to the front."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        if len(system_msgs) > 1:
            # Merge multiple system messages
            combined = " ".join(m.get("content", "") for m in system_msgs)
            system_msgs = [{"role": "system", "content": combined}]
        return system_msgs + other_msgs

    def _remove_orphaned_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove tool results that don't match any tool_call."""
        # Collect all tool_call IDs
        valid_ids = set()
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        valid_ids.add(tc_id)

        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id not in valid_ids:
                    continue  # Skip orphaned
            result.append(msg)
        return result

    def _fix_role_alternation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix consecutive same-role messages by inserting placeholders."""
        if not messages:
            return messages

        result = [messages[0]]
        for i in range(1, len(messages)):
            prev_role = result[-1].get("role")
            curr_role = messages[i].get("role")

            if prev_role == "user" and curr_role == "user":
                # Insert placeholder assistant
                result.append({
                    "role": "assistant",
                    "content": "(continued)"
                })
            elif prev_role == "assistant" and curr_role == "assistant":
                # Insert placeholder user
                result.append({
                    "role": "user",
                    "content": "(continued)"
                })

            result.append(messages[i])
        return result

    def _insert_missing_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert placeholder tool results for unanswered tool_calls."""
        # Find all tool_call IDs and which have results
        tool_call_positions: Dict[str, int] = {}  # id -> position of assistant msg
        answered: set = set()

        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_call_positions[tc_id] = i
            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    answered.add(tc_id)

        missing = {tc_id: pos for tc_id, pos in tool_call_positions.items()
                   if tc_id not in answered}

        if not missing:
            return messages

        # Group missing by assistant position
        by_position: Dict[int, List[str]] = defaultdict(list)
        for tc_id, pos in missing.items():
            by_position[pos].append(tc_id)

        # Insert tool results after the last tool result (or assistant msg)
        result = list(messages)
        offset = 0
        for pos in sorted(by_position.keys()):
            # Find insertion point: after last tool result for this assistant, or after assistant
            insert_at = pos + offset + 1
            while (insert_at < len(result) and
                   result[insert_at].get("role") == "tool"):
                insert_at += 1

            for tc_id in by_position[pos]:
                placeholder = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "(no result available)"
                }
                result.insert(insert_at, placeholder)
                insert_at += 1
                offset += 1

        return result
