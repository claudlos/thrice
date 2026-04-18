"""
Unified Hermes Invariant System.

Merges the two parallel invariant implementations:
  - new-files/hermes_invariants.py  (InvariantChecker, flat violation strings)
  - thrice/invariants/              (BaseInvariantChecker, InvariantError objects)

This module is the SINGLE CANONICAL invariant system. It:
  1. Uses thrice's BaseInvariantChecker pattern (safety/liveness split)
  2. Uses new-files' enforcement.py modes (DEV/PROD/TEST)
  3. Provides a single UnifiedInvariantChecker class with all subsystem methods
  4. Returns List[str] violations for enforcement.py compatibility
  5. Includes deprecation shims for old import paths

Usage:
    from invariants_unified import UnifiedInvariantChecker, enforce

    violations = UnifiedInvariantChecker.check_cron_invariants(jobs)
    enforce(violations, context="cron")

    # Or check everything at once:
    results = UnifiedInvariantChecker.check_all(cron_jobs=jobs, messages=msgs)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Re-export enforcement module for convenience
try:
    from enforcement import (
        EnforcementMode,
        ViolationCollector,
        check_invariants_after,
        enforce,
        get_enforcement_mode,
        get_global_collector,
        is_silent,
        reset_global_collector,
    )
    from enforcement import (
        InvariantViolation as EnforcementViolation,
    )
except ImportError:
    # Stub if enforcement.py not on path
    class EnforcementMode:
        DEV = "dev"
        PROD = "production"
        TEST = "test"

    class EnforcementViolation(Exception):
        pass

    InvariantViolation = EnforcementViolation

    class ViolationCollector:
        def __init__(self):
            self.violations = []
        def add(self, v):
            self.violations.append(v)
        def get_all(self):
            return list(self.violations)
        def clear(self):
            self.violations.clear()

    def check_invariants_after(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def enforce(violations, **kw):
        return violations
    def get_enforcement_mode():
        return "production"

    _global_collector = ViolationCollector()

    def get_global_collector():
        return _global_collector

    def is_silent():
        return False

    def reset_global_collector():
        global _global_collector
        _global_collector = ViolationCollector()


# ---------------------------------------------------------------------------
# Constants — union of both systems
# ---------------------------------------------------------------------------

VALID_ROLES = {"system", "user", "assistant", "tool"}

VALID_CRON_STATES = {
    "nonexistent", "scheduled", "running", "paused",
    "completed", "failed", "removed",
}

VALID_SCHEDULE_KINDS = {"once", "interval", "cron"}
VALID_STATUSES = {None, "ok", "error"}

VALID_PROCESS_STATES = {"spawning", "running", "exiting", "finished", "pruned", "detached"}

VALID_SESSION_STATES = {"created", "active", "idle", "expired", "reset", "interrupted"}

VALID_STREAM_STATES = {"initial", "streaming", "degraded", "done"}

# Cron transition map (from thrice)
_VALID_CRON_TRANSITIONS: Dict[str, Set[str]] = {
    "created":    {"scheduled", "disabled"},
    "scheduled":  {"due", "running", "paused", "disabled", "deleted"},
    "due":        {"running", "disabled", "deleted"},
    "running":    {"scheduled", "completed", "failed"},
    "completed":  {"scheduled", "disabled", "deleted", "removed"},
    "failed":     {"scheduled", "paused", "disabled", "deleted"},
    "paused":     {"scheduled"},
    "disabled":   {"scheduled", "deleted"},
    "deleted":    set(),
    "removed":    set(),
}

# Process transition map (from thrice)
_VALID_PROCESS_TRANSITIONS: Dict[str, Set[str]] = {
    "spawning":  {"running", "finished"},
    "running":   {"exiting", "finished"},
    "exiting":   {"finished"},
    "finished":  {"pruned"},
    "detached":  {"running", "finished"},
    "pruned":    set(),
}

# Session transition map (from thrice)
_VALID_SESSION_TRANSITIONS: Dict[str, Set[str]] = {
    "created":     {"active"},
    "active":      {"idle", "interrupted", "expired"},
    "idle":        {"active", "expired"},
    "expired":     {"reset"},
    "reset":       {"active"},
    "interrupted": {"active", "reset"},
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _get_call_id(tc: Any) -> str:
    """Extract call_id from a tool_call dict."""
    if isinstance(tc, dict):
        return tc.get("id", "") or ""
    if hasattr(tc, "id"):
        return getattr(tc, "id", "") or ""
    return ""


# ---------------------------------------------------------------------------
# Unified Invariant Checker
# ---------------------------------------------------------------------------

class UnifiedInvariantChecker:
    """Single canonical invariant checker for all Hermes subsystems.

    All methods are static and return List[str] violation descriptions.
    Empty list = all invariants hold.

    Merges:
      - hermes_invariants.InvariantChecker (new-files)
      - thrice/invariants/* checkers
    """

    # =====================================================================
    # 1. MESSAGE INVARIANTS (from both systems, union)
    # =====================================================================

    @staticmethod
    def check_message_integrity(messages: List[Dict[str, Any]]) -> List[str]:
        """Verify message list well-formedness.

        Invariants: MI-1..MI-8 (new-files) + INV-M1..M7 (thrice)
        """
        violations: List[str] = []

        if not messages:
            violations.append("EMPTY_MESSAGES: messages list is empty")
            return violations

        all_tool_call_ids: Set[str] = set()
        tool_call_origin: Dict[str, int] = {}  # call_id -> assistant msg index

        for i, msg in enumerate(messages):
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

            # First message role
            if i == 0 and role not in ("system", "user"):
                violations.append(
                    f"BAD_FIRST_MESSAGE[{i}]: role='{role}', expected 'system' or 'user'"
                )

            # Track tool_call IDs
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    seen_in_msg: set = set()
                    for tc in tool_calls:
                        cid = _get_call_id(tc)
                        if cid:
                            if cid in seen_in_msg:
                                violations.append(
                                    f"DUPLICATE_CALL_ID[{i}]: tool_call id '{cid}' "
                                    f"appears twice in same assistant message"
                                )
                            seen_in_msg.add(cid)
                            all_tool_call_ids.add(cid)
                            tool_call_origin[cid] = i

                # Empty assistant
                has_content = bool(msg.get("content"))
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls:
                    violations.append(
                        f"EMPTY_ASSISTANT[{i}]: no content and no tool_calls"
                    )

            # Tool result checks
            if role == "tool" and i > 0:
                prev = messages[i - 1]
                prev_role = prev.get("role") if isinstance(prev, dict) else None

                if prev_role == "assistant":
                    if not prev.get("tool_calls"):
                        violations.append(
                            f"ORPHAN_TOOL_RESULT[{i}]: preceding assistant has no tool_calls"
                        )
                elif prev_role != "tool":
                    violations.append(
                        f"ORPHAN_TOOL_RESULT[{i}]: preceded by '{prev_role}', "
                        f"expected 'assistant' or 'tool'"
                    )

                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    violations.append(
                        f"MISSING_TOOL_CALL_ID[{i}]: tool message has no tool_call_id"
                    )
                elif tool_call_id not in all_tool_call_ids:
                    violations.append(
                        f"UNMATCHED_TOOL_CALL_ID[{i}]: '{tool_call_id}' "
                        f"does not match any preceding tool_call"
                    )
                else:
                    # Ordering check (from thrice INV-M6)
                    origin_idx = tool_call_origin.get(tool_call_id)
                    if origin_idx is not None and i <= origin_idx:
                        violations.append(
                            f"TOOL_RESULT_BEFORE_CALL[{i}]: result for '{tool_call_id}' "
                            f"at index {i} appears before its call at index {origin_idx}"
                        )

            # Consecutive same role (except tool)
            if role == "user" and i > 0:
                prev = messages[i - 1]
                if isinstance(prev, dict) and prev.get("role") == "user":
                    violations.append(
                        f"CONSECUTIVE_USER[{i}]: two user messages in a row"
                    )

        return violations

    # =====================================================================
    # 2. COMPRESSION INVARIANTS (from thrice)
    # =====================================================================

    @staticmethod
    def check_compression_invariants(
        before: List[Dict[str, Any]],
        after: List[Dict[str, Any]],
    ) -> List[str]:
        """Verify compression preserved essential structure.

        Invariants: INV-C1 (progress), INV-C2 (system prompt), INV-C3 (well-formed)
        """
        violations: List[str] = []

        if len(after) >= len(before) and len(before) > 1:
            violations.append(
                f"COMPRESSION_NO_PROGRESS: {len(before)} -> {len(after)} messages"
            )

        before_system = [m for m in before if m.get("role") == "system"]
        after_system = [m for m in after if m.get("role") == "system"]
        if before_system and not after_system:
            violations.append("SYSTEM_PROMPT_LOST: system prompt removed by compression")

        # Post-compression must also be well-formed
        post_violations = UnifiedInvariantChecker.check_message_integrity(after)
        for v in post_violations:
            violations.append(f"POST_COMPRESSION_{v}")

        return violations

    # =====================================================================
    # 3. CRON JOB INVARIANTS (merged — union of both systems)
    # =====================================================================

    @staticmethod
    def check_cron_invariants(jobs: List[Dict[str, Any]]) -> List[str]:
        """Verify cron job list consistency.

        Invariants: CR-1..CR-8 (new-files) + INV-CJ1..CJ9 (thrice)
        """
        violations: List[str] = []
        seen_ids: Set[str] = set()

        for job in jobs:
            jid = job.get("id", job.get("job_id", "?"))

            # Unique IDs (CR-2 / INV-CJ6)
            if jid in seen_ids:
                violations.append(f"DUPLICATE_JOB_ID[{jid}]")
            seen_ids.add(jid)

            # Valid state (CR-1)
            state = job.get("state", "scheduled")
            if state not in VALID_CRON_STATES:
                violations.append(
                    f"INVALID_CRON_STATE[{jid}]: '{state}' not in {VALID_CRON_STATES}"
                )
                continue

            # Valid schedule kind (INV-CJ7)
            schedule = job.get("schedule", {})
            kind = schedule.get("kind", "") if isinstance(schedule, dict) else ""
            if kind and kind not in VALID_SCHEDULE_KINDS:
                violations.append(
                    f"INVALID_SCHEDULE_KIND[{jid}]: '{kind}'"
                )

            # Completed jobs (CR-3)
            if state == "completed" and job.get("next_run_at") is not None:
                violations.append(
                    f"COMPLETED_WITH_NEXT_RUN[{jid}]"
                )

            # Paused + enabled (CR-4)
            if state == "paused" and job.get("enabled", False):
                violations.append(f"PAUSED_BUT_ENABLED[{jid}]")

            # Scheduled without next_run_at (CR-5)
            if (
                state == "scheduled"
                and job.get("enabled", True)
                and job.get("next_run_at") is None
            ):
                violations.append(f"SCHEDULED_NO_NEXT_RUN[{jid}]")

            # Repeat counters (CR-6 / INV-CJ1 / INV-CJ2)
            repeat = job.get("repeat", {})
            if isinstance(repeat, dict) and repeat:
                total = repeat.get("times")
                completed_count = repeat.get("completed", 0)
                remaining = repeat.get("remaining")

                if completed_count < 0:
                    violations.append(
                        f"NEGATIVE_COMPLETED[{jid}]: completed={completed_count}"
                    )

                if total is not None and completed_count > total:
                    violations.append(
                        f"REPEAT_OVERFLOW[{jid}]: completed {completed_count} > limit {total}"
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

            # Running jobs timing (CR-7)
            if state == "running":
                has_timing = (
                    job.get("last_run_at") is not None
                    or job.get("started_at") is not None
                )
                if not has_timing:
                    violations.append(f"RUNNING_NO_TIMING[{jid}]")

            # Removed jobs (CR-8)
            if state == "removed" and job.get("enabled", False):
                violations.append(f"REMOVED_BUT_ENABLED[{jid}]")

            # Status consistency (INV-CJ8)
            status = job.get("last_status")
            if status not in VALID_STATUSES:
                violations.append(
                    f"INVALID_STATUS[{jid}]: '{status}'"
                )
            error = job.get("last_error")
            if status == "error" and not error:
                violations.append(
                    f"ERROR_NO_MESSAGE[{jid}]: last_status='error' but last_error empty"
                )

            # Temporal: last_run_at < next_run_at (INV-CJ4)
            last_run = job.get("last_run_at")
            next_run = job.get("next_run_at")
            if last_run and next_run:
                try:
                    last_dt = datetime.fromisoformat(last_run)
                    next_dt = datetime.fromisoformat(next_run)
                    if last_dt >= next_dt:
                        violations.append(
                            f"LAST_RUN_AFTER_NEXT[{jid}]: {last_run} >= {next_run}"
                        )
                except (ValueError, TypeError):
                    pass

            # Temporal: created_at <= last_run_at (INV-CJ5)
            created = job.get("created_at")
            if created and last_run:
                try:
                    created_dt = datetime.fromisoformat(created)
                    last_run_dt = datetime.fromisoformat(last_run)
                    if created_dt > last_run_dt:
                        violations.append(
                            f"CREATED_AFTER_LAST_RUN[{jid}]: {created} > {last_run}"
                        )
                except (ValueError, TypeError):
                    pass

        return violations

    @staticmethod
    def check_cron_transition(old_state: str, new_state: str) -> List[str]:
        """Validate a cron job state transition."""
        if old_state == new_state:
            return []
        allowed = _VALID_CRON_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            return [
                f"INVALID_CRON_TRANSITION: '{old_state}' -> '{new_state}' "
                f"(allowed: {sorted(allowed)})"
            ]
        return []

    # =====================================================================
    # 4. PROCESS REGISTRY INVARIANTS (merged)
    # =====================================================================

    @staticmethod
    def check_process_registry(
        running: Dict[str, Any],
        finished: Dict[str, Any],
    ) -> List[str]:
        """Verify process registry consistency.

        Invariants: PR-1..PR-5 (new-files) + INV-P1..P9 (thrice)
        """
        violations: List[str] = []

        # Exclusive placement (PR-1 / INV-P1)
        overlap = set(running.keys()) & set(finished.keys())
        if overlap:
            violations.append(
                f"DUAL_REGISTRY: {sorted(overlap)} in both running and finished"
            )

        for sid, session in running.items():
            exited = _safe_getattr(session, "exited", False)
            if exited:
                violations.append(f"RUNNING_BUT_EXITED[{sid}]")

            exit_code = _safe_getattr(session, "exit_code", None)
            if not exited and exit_code is not None:
                violations.append(
                    f"EXIT_CODE_SET_NOT_EXITED[{sid}]: exit_code={exit_code}"
                )

            reader = _safe_getattr(session, "_reader_thread", None)
            if reader is not None and not _safe_getattr(reader, "is_alive", lambda: True)():
                violations.append(f"DEAD_READER[{sid}]")

        for sid, session in finished.items():
            exited = _safe_getattr(session, "exited", True)
            if not exited:
                violations.append(f"FINISHED_NOT_EXITED[{sid}]")

            exit_code = _safe_getattr(session, "exit_code", "MISSING")
            if exit_code == "MISSING":
                pass
            elif exited and exit_code is None:
                violations.append(f"FINISHED_NO_EXIT_CODE[{sid}]")

        return violations

    @staticmethod
    def check_process_transition(old_state: str, new_state: str) -> List[str]:
        """Validate a process state transition."""
        if old_state == new_state:
            return []
        allowed = _VALID_PROCESS_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            return [
                f"INVALID_PROCESS_TRANSITION: '{old_state}' -> '{new_state}' "
                f"(allowed: {sorted(allowed)})"
            ]
        return []

    # =====================================================================
    # 5. SESSION INVARIANTS (merged)
    # =====================================================================

    @staticmethod
    def check_session_invariants(entries: Dict[str, Any]) -> List[str]:
        """Verify gateway session store consistency.

        Invariants: SS-1..SS-5 (new-files) + INV-S1..S5 (thrice)
        """
        violations: List[str] = []
        seen_session_ids: Dict[str, str] = {}

        for key, entry in entries.items():
            session_id = _safe_getattr(entry, "session_id", None)

            if not session_id:
                violations.append(f"MISSING_SESSION_ID[{key}]")
                continue

            if session_id in seen_session_ids:
                violations.append(
                    f"DUPLICATE_SESSION_ID: '{session_id}' used by both "
                    f"'{seen_session_ids[session_id]}' and '{key}'"
                )
            seen_session_ids[session_id] = key

            for token_field in ("input_tokens", "output_tokens", "total_tokens"):
                val = _safe_getattr(entry, token_field, None)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    violations.append(f"NEGATIVE_TOKENS[{key}]: {token_field}={val}")

            auto_reset = _safe_getattr(entry, "was_auto_reset", "MISSING")
            if auto_reset != "MISSING" and not isinstance(auto_reset, bool):
                violations.append(
                    f"INVALID_AUTO_RESET[{key}]: was_auto_reset={auto_reset!r}"
                )

            entry_key = _safe_getattr(entry, "session_key", None)
            if entry_key is not None and entry_key != key:
                violations.append(
                    f"KEY_MISMATCH[{key}]: entry.session_key='{entry_key}' != '{key}'"
                )

        return violations

    @staticmethod
    def check_session_transition(old_state: str, new_state: str) -> List[str]:
        """Validate a session state transition."""
        if old_state == new_state:
            return []
        allowed = _VALID_SESSION_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            return [
                f"INVALID_SESSION_TRANSITION: '{old_state}' -> '{new_state}' "
                f"(allowed: {sorted(allowed)})"
            ]
        return []

    @staticmethod
    def check_monotonic_token_counts(
        old_session: Dict[str, Any],
        new_session: Dict[str, Any],
    ) -> List[str]:
        """INV-S1: Token counts must never decrease."""
        violations: List[str] = []
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            old_val = old_session.get(field, 0) or 0
            new_val = new_session.get(field, 0) or 0
            if new_val < old_val:
                violations.append(
                    f"TOKEN_DECREASE: {field} decreased from {old_val} to {new_val}"
                )
        return violations

    # =====================================================================
    # 6. DELEGATION TREE INVARIANTS (from new-files, unique)
    # =====================================================================

    @staticmethod
    def check_delegation_tree(
        parent_agent: Any,
        active_children: List[Any],
        depth: int,
        max_depth: int = 2,
        max_children: int = 3,
    ) -> List[str]:
        """Verify delegation tree consistency.

        Invariants: DT-1..DT-5
        """
        violations: List[str] = []

        if depth > max_depth:
            violations.append(f"DEPTH_EXCEEDED: {depth} > max {max_depth}")

        if len(active_children) > max_children:
            violations.append(
                f"TOO_MANY_CHILDREN: {len(active_children)} (max {max_children})"
            )

        child_ids = [id(c) for c in active_children]
        if len(child_ids) != len(set(child_ids)):
            violations.append("DUPLICATE_CHILD: same child object appears twice")

        parent_budget = _safe_getattr(parent_agent, "iteration_budget", None)
        if parent_budget is not None:
            total_consumed = sum(
                _safe_getattr(c, "iterations_consumed", 0) or 0
                for c in active_children
            )
            if total_consumed > parent_budget:
                violations.append(
                    f"BUDGET_EXCEEDED: children consumed {total_consumed} "
                    f"but parent budget is {parent_budget}"
                )

        for i, child in enumerate(active_children):
            child_parent = _safe_getattr(child, "parent_agent", None)
            if child_parent is not None and child_parent is not parent_agent:
                violations.append(f"WRONG_PARENT[child_{i}]")

        return violations

    # =====================================================================
    # 7. STREAM CONSUMER INVARIANTS (from new-files, unique)
    # =====================================================================

    @staticmethod
    def check_stream_state(consumer: Any) -> List[str]:
        """Verify stream consumer state consistency.

        Invariants: SC-1..SC-5
        """
        violations: List[str] = []

        state = _safe_getattr(consumer, "_state", None)
        if state is None:
            return violations

        state_name = _safe_getattr(state, "name", str(state))

        if state_name in ("DONE", "done"):
            queue = _safe_getattr(consumer, "_queue", None)
            if queue is not None:
                try:
                    if not queue.empty():
                        violations.append("DONE_WITH_PENDING: queue not empty after DONE")
                except Exception:
                    pass

        if state_name in ("DEGRADED", "degraded"):
            edit_supported = _safe_getattr(consumer, "_edit_supported", None)
            if edit_supported is True:
                violations.append("DEGRADED_EDIT_MISMATCH: DEGRADED but _edit_supported=True")

        finish_count = _safe_getattr(consumer, "_finish_count", None)
        if finish_count is not None and isinstance(finish_count, int) and finish_count > 1:
            violations.append(f"MULTIPLE_FINISH: finish() called {finish_count} times")

        edit_ever_disabled = _safe_getattr(consumer, "_edit_ever_disabled", None)
        edit_supported = _safe_getattr(consumer, "_edit_supported", None)
        if edit_ever_disabled is True and edit_supported is True:
            violations.append("EDIT_LATCH_VIOLATION: _edit_supported re-enabled after disable")

        return violations

    # =====================================================================
    # 8. TURN INVARIANTS (from thrice, unique)
    # =====================================================================

    @staticmethod
    def check_turn_completeness(turn_messages: List[Dict[str, Any]]) -> List[str]:
        """Verify turn tool-call/result completeness.

        Invariants: INV-T1, INV-T4
        """
        violations: List[str] = []
        if not turn_messages:
            return violations

        assistant_msg = turn_messages[0]
        if assistant_msg.get("role") != "assistant":
            return violations

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            return violations

        expected_ids: Set[str] = set()
        seen_call_ids: set = set()
        for tc in tool_calls:
            cid = _get_call_id(tc)
            if cid:
                if cid in seen_call_ids:
                    violations.append(f"DUPLICATE_CALL_ID_IN_TURN: '{cid}'")
                seen_call_ids.add(cid)
                expected_ids.add(cid)

        result_ids: List[str] = []
        for msg in turn_messages[1:]:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id", "")
                if cid:
                    result_ids.append(cid)

        result_set = set(result_ids)
        missing = expected_ids - result_set
        if missing:
            violations.append(
                f"MISSING_TURN_RESULTS: {len(missing)} call(s) without results: "
                f"{', '.join(sorted(missing)[:5])}"
            )

        extra = result_set - expected_ids
        if extra:
            violations.append(
                f"EXTRA_TURN_RESULTS: {len(extra)} result(s) without matching calls"
            )

        return violations

    @staticmethod
    def check_turn_boundary(
        pre_turn_messages: List[Dict[str, Any]],
        post_turn_messages: List[Dict[str, Any]],
    ) -> List[str]:
        """INV-T3: pre-turn must be a prefix of post-turn."""
        violations: List[str] = []

        if len(post_turn_messages) < len(pre_turn_messages):
            violations.append(
                f"TURN_BOUNDARY_SHRUNK: {len(pre_turn_messages)} -> {len(post_turn_messages)}"
            )
            return violations

        for i in range(len(pre_turn_messages)):
            if pre_turn_messages[i] != post_turn_messages[i]:
                violations.append(
                    f"TURN_HISTORY_MUTATED[{i}]: message at index {i} was changed"
                )
                break

        return violations

    # =====================================================================
    # AGGREGATE CHECK
    # =====================================================================

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
        """Run all applicable checks. Returns dict of subsystem -> violations."""
        results: Dict[str, List[str]] = {}

        if messages is not None:
            results["messages"] = cls.check_message_integrity(messages)

        if cron_jobs is not None:
            results["cron"] = cls.check_cron_invariants(cron_jobs)

        if running_processes is not None or finished_processes is not None:
            results["processes"] = cls.check_process_registry(
                running_processes or {}, finished_processes or {}
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
                delegation_parent, delegation_children, delegation_depth
            )

        return results

    @classmethod
    def check_all_flat(cls, **kwargs) -> List[str]:
        """Like check_all but returns flat list with subsystem prefixes."""
        results = cls.check_all(**kwargs)
        flat: List[str] = []
        for subsystem, violations in results.items():
            for v in violations:
                flat.append(f"[{subsystem}] {v}")
        return flat


# ---------------------------------------------------------------------------
# Deprecation shims — old import paths still work
# ---------------------------------------------------------------------------

# Alias so `from hermes_invariants import InvariantChecker` still works
InvariantChecker = UnifiedInvariantChecker

# Alias so `from invariants_unified import check_cron_invariants` works at module level
check_message_integrity = UnifiedInvariantChecker.check_message_integrity
check_cron_invariants = UnifiedInvariantChecker.check_cron_invariants
check_process_registry = UnifiedInvariantChecker.check_process_registry
check_session_invariants = UnifiedInvariantChecker.check_session_invariants
check_delegation_tree = UnifiedInvariantChecker.check_delegation_tree
check_stream_state = UnifiedInvariantChecker.check_stream_state
check_turn_completeness = UnifiedInvariantChecker.check_turn_completeness
check_compression_invariants = UnifiedInvariantChecker.check_compression_invariants
