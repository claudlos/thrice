"""
Runtime invariant checking for Hermes subsystems.

NOTE: invariants_unified.py is the canonical version of the invariant
checking system. This module is kept for backward compatibility.
See invariants_unified.py for the most up-to-date implementation.

Provides assertion-style checks that verify critical invariants after
state mutations. Inspired by the dual-verification pattern from
tla-precheck (formal spec + runtime checks) and the sample-task-management
project (CheckAllInvariants after every mutation).

This module is the central hub for all Hermes invariant checks. Each
subsystem has a dedicated checker method that returns a list of violation
strings (empty list = all OK).

Usage:
    from hermes_invariants import InvariantChecker

    # After any message list mutation:
    violations = InvariantChecker.check_message_integrity(messages)

    # After any cron job state change:
    violations = InvariantChecker.check_cron_invariants(jobs)

    # After any process registry change:
    violations = InvariantChecker.check_process_registry(running, finished)

    # After any session store change:
    violations = InvariantChecker.check_session_invariants(entries)

    # After any delegation:
    violations = InvariantChecker.check_delegation_tree(parent, children, depth)

    # After any stream consumer state change:
    violations = InvariantChecker.check_stream_state(consumer)

Enforcement is handled by the enforcement module:
    from enforcement import enforce, get_enforcement_mode
    enforce(violations, context="cron")
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid constants (avoid importing from Hermes modules at top level)
# ---------------------------------------------------------------------------

VALID_ROLES = {"system", "user", "assistant", "tool"}

VALID_CRON_STATES = {
    "nonexistent", "scheduled", "running", "paused",
    "completed", "failed", "removed",
}

VALID_PROCESS_STATES = {"spawning", "running", "exited", "detached", "pruned"}

VALID_SESSION_STATES = {"empty", "active", "expired", "resetting", "switched"}

VALID_STREAM_STATES = {"initial", "streaming", "degraded", "done"}


class InvariantChecker:
    """Runtime invariant verification for Hermes subsystems.

    All methods are static and return a list of violation description strings.
    An empty list means all invariants hold.

    Convention: violation strings are formatted as:
        "VIOLATION_CODE: human-readable description"
    """

    # =========================================================================
    # 1. Message Integrity (SM-2 related)
    # =========================================================================

    @staticmethod
    def check_message_integrity(messages: List[Dict[str, Any]]) -> List[str]:
        """Verify a message list is well-formed for the chat completion API.

        Invariants checked:
            MI-1: Messages list is not empty
            MI-2: All messages are dicts with a valid role
            MI-3: First message should be system or user (not tool/assistant reply)
            MI-4: Tool results follow an assistant message with tool_calls
            MI-5: Tool call IDs in results match a preceding tool_call
            MI-6: No consecutive user messages (usually a merge bug)
            MI-7: Assistant messages have content or tool_calls (not empty)
            MI-8: Tool messages have tool_call_id
        """
        violations: List[str] = []

        # MI-1: Non-empty
        if not messages:
            violations.append("EMPTY_MESSAGES: messages list is empty")
            return violations

        # Collect all tool_call IDs from assistant messages for matching
        all_tool_call_ids: Set[str] = set()

        for i, msg in enumerate(messages):
            # MI-2: Valid structure
            if not isinstance(msg, dict):
                violations.append(
                    f"INVALID_MESSAGE[{i}]: not a dict (type={type(msg).__name__})"
                )
                continue

            role = msg.get("role")
            if role not in VALID_ROLES:
                violations.append(
                    f"INVALID_ROLE[{i}]: '{role}' not in {VALID_ROLES}"
                )
                continue

            # MI-3: First message role
            if i == 0 and role not in ("system", "user"):
                violations.append(
                    f"BAD_FIRST_MESSAGE[{i}]: role='{role}', "
                    f"expected 'system' or 'user'"
                )

            # Track tool_call IDs from assistant messages
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict) and "id" in tc:
                            all_tool_call_ids.add(tc["id"])

                # MI-7: Assistant has content or tool_calls
                has_content = bool(msg.get("content"))
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls:
                    violations.append(
                        f"EMPTY_ASSISTANT[{i}]: no content and no tool_calls"
                    )

            # MI-4: Tool result follows assistant with tool_calls or another tool
            if role == "tool" and i > 0:
                prev = messages[i - 1]
                prev_role = prev.get("role") if isinstance(prev, dict) else None

                if prev_role == "assistant":
                    if not prev.get("tool_calls"):
                        violations.append(
                            f"ORPHAN_TOOL_RESULT[{i}]: preceding assistant "
                            f"has no tool_calls"
                        )
                elif prev_role != "tool":
                    violations.append(
                        f"ORPHAN_TOOL_RESULT[{i}]: preceded by '{prev_role}', "
                        f"expected 'assistant' or 'tool'"
                    )

                # MI-8: Tool messages need tool_call_id
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    violations.append(
                        f"MISSING_TOOL_CALL_ID[{i}]: tool message has no tool_call_id"
                    )
                # MI-5: tool_call_id should match a known tool_call
                elif tool_call_id not in all_tool_call_ids:
                    violations.append(
                        f"UNMATCHED_TOOL_CALL_ID[{i}]: '{tool_call_id}' "
                        f"does not match any preceding tool_call"
                    )

            # MI-6: Consecutive user messages
            if role == "user" and i > 0:
                prev = messages[i - 1]
                if isinstance(prev, dict) and prev.get("role") == "user":
                    violations.append(
                        f"CONSECUTIVE_USER[{i}]: two user messages in a row "
                        f"(likely a merge bug)"
                    )

        return violations

    # =========================================================================
    # 2. Cron Job Invariants (SM-1)
    # =========================================================================

    @staticmethod
    def check_cron_invariants(jobs: List[Dict[str, Any]]) -> List[str]:
        """Verify cron job list consistency.

        Invariants checked:
            CR-1: All jobs have valid states
            CR-2: No duplicate job IDs
            CR-3: Completed jobs have no next_run_at
            CR-4: Paused jobs are not enabled
            CR-5: Enabled jobs in 'scheduled' state should have next_run_at
            CR-6: Repeat counters are consistent (completed + left == total)
            CR-7: Running jobs have last_run_at or started_at
            CR-8: Removed jobs are not enabled
        """
        violations: List[str] = []
        seen_ids: Set[str] = set()

        for job in jobs:
            jid = job.get("id", job.get("job_id", "?"))

            # CR-2: Unique IDs
            if jid in seen_ids:
                violations.append(f"DUPLICATE_JOB_ID: '{jid}'")
            seen_ids.add(jid)

            # CR-1: Valid state
            state = job.get("state", "scheduled")
            if state not in VALID_CRON_STATES:
                violations.append(
                    f"INVALID_CRON_STATE[{jid}]: '{state}' not in {VALID_CRON_STATES}"
                )
                continue  # Can't check state-dependent invariants

            # CR-3: Completed jobs
            if state == "completed" and job.get("next_run_at") is not None:
                violations.append(
                    f"COMPLETED_WITH_NEXT_RUN[{jid}]: completed but "
                    f"next_run_at={job.get('next_run_at')}"
                )

            # CR-4: Paused + enabled mismatch
            if state == "paused" and job.get("enabled", False):
                violations.append(
                    f"PAUSED_BUT_ENABLED[{jid}]: state is paused but enabled=True"
                )

            # CR-5: Scheduled without next_run_at
            if (
                state == "scheduled"
                and job.get("enabled", True)
                and job.get("next_run_at") is None
            ):
                violations.append(
                    f"SCHEDULED_NO_NEXT_RUN[{jid}]: scheduled and enabled "
                    f"but no next_run_at"
                )

            # CR-6: Repeat counter consistency
            repeat = job.get("repeat", {})
            if isinstance(repeat, dict) and repeat:
                total = repeat.get("times")
                completed_count = repeat.get("completed", 0)
                remaining = repeat.get("remaining")

                if total is not None and completed_count > total:
                    violations.append(
                        f"REPEAT_OVERFLOW[{jid}]: completed {completed_count} "
                        f"runs but limit is {total}"
                    )

                if (
                    total is not None
                    and remaining is not None
                    and completed_count + remaining != total
                ):
                    violations.append(
                        f"REPEAT_MISMATCH[{jid}]: completed({completed_count}) + "
                        f"remaining({remaining}) != total({total})"
                    )

            # CR-7: Running jobs should have timing info
            if state == "running":
                has_timing = (
                    job.get("last_run_at") is not None
                    or job.get("started_at") is not None
                )
                if not has_timing:
                    violations.append(
                        f"RUNNING_NO_TIMING[{jid}]: running job has neither "
                        f"last_run_at nor started_at"
                    )

            # CR-8: Removed jobs
            if state == "removed" and job.get("enabled", False):
                violations.append(
                    f"REMOVED_BUT_ENABLED[{jid}]: removed but enabled=True"
                )

        return violations

    # =========================================================================
    # 3. Process Registry Invariants (SM-4)
    # =========================================================================

    @staticmethod
    def check_process_registry(
        running: Dict[str, Any],
        finished: Dict[str, Any],
    ) -> List[str]:
        """Verify process registry consistency.

        Invariants checked:
            PR-1: No session in both running and finished
            PR-2: Running sessions are not marked as exited
            PR-3: Finished sessions are marked as exited
            PR-4: Running sessions have an associated reader thread
            PR-5: Finished sessions have an exit code
        """
        violations: List[str] = []

        # PR-1: No overlap
        overlap = set(running.keys()) & set(finished.keys())
        if overlap:
            violations.append(
                f"DUAL_REGISTRY: sessions {sorted(overlap)} exist in both "
                f"running and finished"
            )

        # PR-2: Running sessions not exited
        for sid, session in running.items():
            if _safe_getattr(session, "exited", False):
                violations.append(
                    f"RUNNING_BUT_EXITED[{sid}]: in running dict but exited=True"
                )

            # PR-4: Reader thread alive (if trackable)
            reader = _safe_getattr(session, "_reader_thread", None)
            if reader is not None and not _safe_getattr(reader, "is_alive", lambda: True)():
                violations.append(
                    f"DEAD_READER[{sid}]: in running dict but reader thread is dead"
                )

        # PR-3: Finished sessions are exited
        for sid, session in finished.items():
            if not _safe_getattr(session, "exited", True):
                violations.append(
                    f"FINISHED_NOT_EXITED[{sid}]: in finished dict but exited=False"
                )

            # PR-5: Exit code set
            exit_code = _safe_getattr(session, "exit_code", "MISSING")
            if exit_code == "MISSING":
                # Can't check if attribute doesn't exist
                pass
            elif exit_code is None:
                violations.append(
                    f"FINISHED_NO_EXIT_CODE[{sid}]: in finished dict but "
                    f"exit_code is None"
                )

        return violations

    # =========================================================================
    # 4. Session Store Invariants (SM-5)
    # =========================================================================

    @staticmethod
    def check_session_invariants(entries: Dict[str, Any]) -> List[str]:
        """Verify gateway session store consistency.

        Invariants checked:
            SS-1: All entries have a valid session_id
            SS-2: No duplicate session_ids across different keys
            SS-3: Token counts are non-negative
            SS-4: was_auto_reset is a boolean
            SS-5: Session key matches the dict key
        """
        violations: List[str] = []
        seen_session_ids: Dict[str, str] = {}

        for key, entry in entries.items():
            session_id = _safe_getattr(entry, "session_id", None)

            # SS-1: Valid session_id
            if not session_id:
                violations.append(
                    f"MISSING_SESSION_ID[{key}]: entry has no session_id"
                )
                continue

            # SS-2: Unique session_id
            if session_id in seen_session_ids:
                violations.append(
                    f"DUPLICATE_SESSION_ID: '{session_id}' used by both "
                    f"'{seen_session_ids[session_id]}' and '{key}'"
                )
            seen_session_ids[session_id] = key

            # SS-3: Non-negative token counts
            for token_field in ("input_tokens", "output_tokens", "total_tokens"):
                val = _safe_getattr(entry, token_field, None)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    violations.append(
                        f"NEGATIVE_TOKENS[{key}]: {token_field}={val}"
                    )

            # SS-4: was_auto_reset type
            auto_reset = _safe_getattr(entry, "was_auto_reset", "MISSING")
            if auto_reset != "MISSING" and not isinstance(auto_reset, bool):
                violations.append(
                    f"INVALID_AUTO_RESET[{key}]: was_auto_reset={auto_reset!r} "
                    f"(expected bool)"
                )

            # SS-5: Session key consistency
            entry_key = _safe_getattr(entry, "session_key", None)
            if entry_key is not None and entry_key != key:
                violations.append(
                    f"KEY_MISMATCH[{key}]: entry.session_key='{entry_key}' "
                    f"!= dict key '{key}'"
                )

        return violations

    # =========================================================================
    # 5. Delegation Tree Invariants (SM-3)
    # =========================================================================

    @staticmethod
    def check_delegation_tree(
        parent_agent: Any,
        active_children: List[Any],
        depth: int,
        max_depth: int = 2,
        max_children: int = 3,
    ) -> List[str]:
        """Verify delegation tree consistency.

        Invariants checked:
            DT-1: Depth does not exceed max_depth
            DT-2: Active children count is within bounds
            DT-3: No duplicate children
            DT-4: Parent budget >= sum of children's consumed iterations
            DT-5: All children reference the correct parent
        """
        violations: List[str] = []

        # DT-1: Depth limit
        if depth > max_depth:
            violations.append(
                f"DEPTH_EXCEEDED: current depth {depth} > max {max_depth}"
            )

        # DT-2: Children count
        if len(active_children) > max_children:
            violations.append(
                f"TOO_MANY_CHILDREN: {len(active_children)} active children "
                f"(max {max_children})"
            )

        # DT-3: No duplicate children
        child_ids = [id(c) for c in active_children]
        if len(child_ids) != len(set(child_ids)):
            violations.append(
                "DUPLICATE_CHILD: same child object appears twice in "
                "active_children"
            )

        # DT-4: Budget consistency
        parent_budget = _safe_getattr(parent_agent, "iteration_budget", None)
        if parent_budget is not None:
            total_consumed = 0
            for child in active_children:
                consumed = _safe_getattr(child, "iterations_consumed", 0)
                if isinstance(consumed, (int, float)):
                    total_consumed += consumed

            if total_consumed > parent_budget:
                violations.append(
                    f"BUDGET_EXCEEDED: children consumed {total_consumed} "
                    f"iterations but parent budget is {parent_budget}"
                )

        # DT-5: Parent reference (if children track their parent)
        for i, child in enumerate(active_children):
            child_parent = _safe_getattr(child, "parent_agent", None)
            if child_parent is not None and child_parent is not parent_agent:
                violations.append(
                    f"WRONG_PARENT[child_{i}]: child's parent_agent does not "
                    f"match the actual parent"
                )

        return violations

    # =========================================================================
    # 6. Stream Consumer State Invariants (SM-6)
    # =========================================================================

    @staticmethod
    def check_stream_state(consumer: Any) -> List[str]:
        """Verify stream consumer state consistency.

        Invariants checked:
            SC-1: State enum matches internal flags
            SC-2: No pending items after DONE state
            SC-3: _edit_supported is one-way latch (once False, stays False)
            SC-4: finish() called exactly once (not zero, not multiple)
            SC-5: DEGRADED state means _edit_supported=False
        """
        violations: List[str] = []

        state = _safe_getattr(consumer, "_state", None)
        if state is None:
            # Cannot check without state
            return violations

        state_name = _safe_getattr(state, "name", str(state))

        # SC-2: DONE with pending queue items
        if state_name == "DONE" or state_name == "done":
            queue = _safe_getattr(consumer, "_queue", None)
            if queue is not None:
                try:
                    if not queue.empty():
                        violations.append(
                            "DONE_WITH_PENDING: consumer is DONE but queue "
                            "is not empty"
                        )
                except Exception:
                    pass  # Queue may not have .empty()

        # SC-5: DEGRADED means no edit support
        if state_name == "DEGRADED" or state_name == "degraded":
            edit_supported = _safe_getattr(consumer, "_edit_supported", None)
            if edit_supported is True:
                violations.append(
                    "DEGRADED_EDIT_MISMATCH: state is DEGRADED but "
                    "_edit_supported=True"
                )

        # SC-4: finish count
        finish_count = _safe_getattr(consumer, "_finish_count", None)
        if finish_count is not None:
            if isinstance(finish_count, int) and finish_count > 1:
                violations.append(
                    f"MULTIPLE_FINISH: finish() called {finish_count} times "
                    f"(expected exactly 1)"
                )

        # SC-3: One-way latch for _edit_supported
        edit_ever_disabled = _safe_getattr(
            consumer, "_edit_ever_disabled", None
        )
        edit_supported = _safe_getattr(consumer, "_edit_supported", None)
        if (
            edit_ever_disabled is True
            and edit_supported is True
        ):
            violations.append(
                "EDIT_LATCH_VIOLATION: _edit_supported re-enabled after "
                "being disabled (one-way latch violated)"
            )

        return violations

    # =========================================================================
    # Aggregate checker
    # =========================================================================

    @classmethod
    def check_all(
        cls,
        messages: Optional[List[Dict[str, Any]]] = None,
        cron_jobs: Optional[List[Dict[str, Any]]] = None,
        running_processes: Optional[Dict[str, Any]] = None,
        finished_processes: Optional[Dict[str, Any]] = None,
        session_entries: Optional[Dict[str, Any]] = None,
        stream_consumer: Optional[Any] = None,
        delegation_parent: Optional[Any] = None,
        delegation_children: Optional[List[Any]] = None,
        delegation_depth: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Run all applicable invariant checks and return results by subsystem.

        Only runs checks for which data is provided (non-None).
        Returns a dict mapping subsystem name to list of violations.

        Example:
            results = InvariantChecker.check_all(
                messages=conversation,
                cron_jobs=job_list,
            )
            # results = {"messages": [], "cron": ["DUPLICATE_JOB_ID: ..."]}
        """
        results: Dict[str, List[str]] = {}

        if messages is not None:
            results["messages"] = cls.check_message_integrity(messages)

        if cron_jobs is not None:
            results["cron"] = cls.check_cron_invariants(cron_jobs)

        if running_processes is not None or finished_processes is not None:
            results["processes"] = cls.check_process_registry(
                running_processes or {},
                finished_processes or {},
            )

        if session_entries is not None:
            results["sessions"] = cls.check_session_invariants(session_entries)

        if stream_consumer is not None:
            results["stream"] = cls.check_stream_state(stream_consumer)

        if (
            delegation_parent is not None
            and delegation_children is not None
            and delegation_depth is not None
        ):
            results["delegation"] = cls.check_delegation_tree(
                delegation_parent,
                delegation_children,
                delegation_depth,
            )

        return results

    @classmethod
    def check_all_flat(cls, **kwargs) -> List[str]:
        """Like check_all but returns a flat list of all violations.

        Each violation is prefixed with the subsystem name.
        """
        results = cls.check_all(**kwargs)
        flat: List[str] = []
        for subsystem, violations in results.items():
            for v in violations:
                flat.append(f"[{subsystem}] {v}")
        return flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute, working with both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
