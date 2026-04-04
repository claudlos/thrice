"""Debugging-specific system prompt guidance for Hermes Agent.

Provides research-backed debugging instructions and coding best practices
that can be injected into the system prompt to improve agent performance
when errors or failures are detected in the conversation.

Integration point: agent/prompt_builder.py or run_agent.py _build_system_prompt()
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Debugging guidance — injected when errors/failures are detected
# ---------------------------------------------------------------------------

DEBUGGING_GUIDANCE = (
    "# Debugging guidance\n"
    "You are debugging an issue. Follow these research-backed steps:\n"
    "\n"
    "1. **Read the full error trace** before acting — the root cause is often "
    "at the bottom of the traceback, not the top. Do not skim.\n"
    "\n"
    "2. **Search related code before editing** — understand the surrounding "
    "context (imports, callers, data flow) before making changes. Use search "
    "tools to find all references.\n"
    "\n"
    "3. **Make the smallest possible fix** — target the exact line or expression "
    "that causes the error. Resist the urge to refactor while debugging.\n"
    "\n"
    "4. **Run tests after every change** — verify the fix works AND that "
    "existing tests still pass. Never assume a fix is correct without running it.\n"
    "\n"
    "5. **Step back after 3 failed attempts** — if your fix hasn't worked after "
    "three tries, STOP. Re-read the original error from scratch, reconsider your "
    "assumptions, and try a fundamentally different approach.\n"
    "\n"
    "6. **Check imports and dependencies first** — many errors stem from missing "
    "imports, wrong module paths, or version mismatches. Rule these out before "
    "attempting complex fixes.\n"
    "\n"
    "7. **Verify the fix doesn't break existing tests** — run the full relevant "
    "test suite, not just the failing test. A fix that breaks other tests is not "
    "a fix.\n"
)

# ---------------------------------------------------------------------------
# Coding best practices — always-on lightweight guidance
# ---------------------------------------------------------------------------

CODING_BEST_PRACTICES = (
    "# Coding best practices\n"
    "- **Read before write** — always read the file before editing it. Never "
    "edit a file based on assumptions about its contents.\n"
    "- **Verify assumptions** — don't assume file structure, function signatures, "
    "or variable names. Check them with search or read tools first.\n"
    "- **One change at a time** — make atomic edits. Each change should do one "
    "thing. Multiple simultaneous changes are harder to debug when they fail.\n"
)

# ---------------------------------------------------------------------------
# Error/failure detection patterns
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = [
    # Python tracebacks
    re.compile(r'Traceback \(most recent call last\)', re.IGNORECASE),
    re.compile(r'(?:^|\n)\w*Error: .+', re.MULTILINE),
    re.compile(r'(?:^|\n)\w*Exception: .+', re.MULTILINE),
    # Test failures
    re.compile(r'FAILED\s+[\w/\.:]+', re.IGNORECASE),
    re.compile(r'FAIL:\s+test_\w+', re.IGNORECASE),
    re.compile(r'AssertionError', re.IGNORECASE),
    re.compile(r'\d+ (?:failed|error)', re.IGNORECASE),
    # Generic error markers
    re.compile(r'(?:command|exit).{0,20}(?:failed|error|non-zero)', re.IGNORECASE),
    re.compile(r'error\[E\d+\]', re.IGNORECASE),  # Rust-style errors
    re.compile(r'(?:TypeError|ValueError|KeyError|ImportError|ModuleNotFoundError|AttributeError|NameError|SyntaxError|IndentationError|FileNotFoundError|PermissionError|OSError|RuntimeError|NotImplementedError)', re.IGNORECASE),
    # Build failures
    re.compile(r'BUILD FAILED|compilation error|linker error', re.IGNORECASE),
    # JS/TS errors
    re.compile(r'ReferenceError|SyntaxError|TypeError', re.IGNORECASE),
    re.compile(r'Cannot find module', re.IGNORECASE),
]

# Patterns that indicate repeated failure (escalation trigger)
_REPEATED_FAILURE_PATTERNS = [
    re.compile(r'still (?:failing|broken|not working)', re.IGNORECASE),
    re.compile(r'same error', re.IGNORECASE),
    re.compile(r'didn\'t (?:work|fix|help)', re.IGNORECASE),
    re.compile(r'not fixed', re.IGNORECASE),
    re.compile(r'again|another attempt', re.IGNORECASE),
]

# Threshold: how many recent messages to scan
_SCAN_WINDOW = 10


def _extract_text_from_message(message: dict) -> str:
    """Extract plain text content from a message dict.

    Handles both string content and list-of-blocks content formats
    used by various LLM APIs.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # Tool results can contain error output
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        parts.append(result_content)
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(sub.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content else ""


def should_inject_debugging_guidance(messages: List[dict]) -> bool:
    """Detect whether errors or failures are present in recent messages.

    Scans the last _SCAN_WINDOW messages for error patterns. Returns True
    if at least one error pattern is found, indicating the agent is likely
    in a debugging context.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        True if debugging guidance should be injected.
    """
    if not messages:
        return False

    recent = messages[-_SCAN_WINDOW:]
    for msg in recent:
        text = _extract_text_from_message(msg)
        if not text:
            continue
        for pattern in _ERROR_PATTERNS:
            if pattern.search(text):
                return True
    return False


def count_recent_errors(messages: List[dict]) -> int:
    """Count how many of the recent messages contain error patterns.

    Useful for escalation logic — if errors appear in 3+ of the last
    messages, the agent may be stuck in a loop.
    """
    if not messages:
        return 0

    count = 0
    recent = messages[-_SCAN_WINDOW:]
    for msg in recent:
        text = _extract_text_from_message(msg)
        if not text:
            continue
        for pattern in _ERROR_PATTERNS:
            if pattern.search(text):
                count += 1
                break  # Count each message only once
    return count


def is_repeated_failure(messages: List[dict]) -> bool:
    """Detect if the user or system is reporting repeated failures.

    Returns True if repeated failure language is detected, suggesting
    the agent should step back and reconsider its approach.
    """
    if not messages:
        return False

    recent = messages[-5:]
    for msg in recent:
        text = _extract_text_from_message(msg)
        if not text:
            continue
        for pattern in _REPEATED_FAILURE_PATTERNS:
            if pattern.search(text):
                return True

    # Also check if errors appear in 3+ consecutive recent messages
    return count_recent_errors(messages) >= 3


def get_debugging_prompt(messages: Optional[List[dict]] = None) -> str:
    """Return the appropriate debugging guidance based on conversation state.

    If messages are provided, returns escalated guidance when repeated
    failures are detected. Always includes CODING_BEST_PRACTICES.

    Args:
        messages: Optional list of message dicts for context-aware guidance.

    Returns:
        Combined debugging + best practices prompt string.
    """
    parts = [CODING_BEST_PRACTICES]

    if messages is not None and should_inject_debugging_guidance(messages):
        parts.insert(0, DEBUGGING_GUIDANCE)

        if is_repeated_failure(messages):
            parts.append(
                "# ⚠️ Repeated failure detected\n"
                "You appear to be stuck in a debugging loop. STOP and:\n"
                "1. Re-read the ORIGINAL error message from the very beginning\n"
                "2. List your assumptions — which ones have you NOT verified?\n"
                "3. Try a completely different approach\n"
                "4. Consider: is the bug where you think it is, or upstream?\n"
                "5. Check if the test itself is wrong, not just the code\n"
            )

    return "\n".join(parts)

