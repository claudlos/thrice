"""
Cron Job State Machine (SM-1) for Hermes.

Implements the explicit cron job lifecycle state machine defined in the
improvement plan. Every cron job gets its own CronJobStateMachine instance
that enforces valid transitions and invariants.

States: nonexistent, scheduled, running, paused, completed, failed, removed

Usage:
    from cron_state_machine import CronJobStateMachine

    sm = CronJobStateMachine(job_id="daily-backup")
    sm.create(next_run_at=datetime(...), repeat_total=5)
    sm.tick(now=datetime.now())     # scheduled -> running
    sm.mark_success()               # running -> scheduled (if recurring)
    sm.pause()                      # scheduled -> paused
    sm.resume()                     # paused -> scheduled
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from state_machine import (
    GuardFailed,
    InvalidTransition,
    StateMachine,
    TransitionDef,
    TransitionRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cron job states
# ---------------------------------------------------------------------------

# All valid states for SM-1
CRON_STATES = {
    "nonexistent",
    "scheduled",
    "running",
    "paused",
    "completed",
    "failed",
    "removed",
}

# Terminal states (no transitions out except force_state)
TERMINAL_STATES = {"completed", "removed"}

# States from which remove() is allowed
REMOVABLE_STATES = {"scheduled", "paused", "completed", "failed", "nonexistent"}


# ---------------------------------------------------------------------------
# Guard functions
# ---------------------------------------------------------------------------

def _guard_tick(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for tick: job must be enabled and due."""
    enabled = ctx.get("enabled", True)
    now = ctx.get("now")
    next_run_at = ctx.get("next_run_at")

    if not enabled:
        return False
    if now is None or next_run_at is None:
        return False
    if isinstance(now, datetime) and isinstance(next_run_at, datetime):
        return now >= next_run_at
    # Fallback: allow if both are provided (caller pre-checked)
    return True


def _guard_mark_success_recurring(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for mark_success -> scheduled: recurring with strictly more than
    one run left (this run plus at least one more).  Matches TLA+
    ``MarkSuccessRecurring`` precondition ``runsLeft > 1``; keeps the
    ``RepeatConsistency`` invariant by routing the final bounded run through
    ``mark_success_completed`` instead of a scheduled-but-exhausted state."""
    recurring = ctx.get("recurring", False)
    runs_left = ctx.get("runs_left")

    if not recurring:
        return False
    if runs_left is not None and runs_left <= 1:
        return False
    return True


def _guard_mark_success_completed(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for mark_success -> completed: one-shot, or the last run of a
    bounded recurring schedule.  Matches TLA+ ``MarkSuccessDone`` precondition
    ``~recurring \\/ runsLeft <= 1``."""
    recurring = ctx.get("recurring", False)
    runs_left = ctx.get("runs_left")

    if not recurring:
        return True
    if runs_left is not None and runs_left <= 1:
        return True
    return False


def _guard_retry(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for retry: retry count must be below max."""
    retry_count = ctx.get("retry_count", 0)
    max_retries = ctx.get("max_retries", 3)
    return retry_count < max_retries


def _guard_remove(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for remove: cannot remove a running job."""
    return current != "running"


def _guard_recover(current: str, action: str, ctx: Dict[str, Any]) -> bool:
    """Guard for recover: stale lock must be detected."""
    return ctx.get("stale_lock", False)


# ---------------------------------------------------------------------------
# Invariant functions
# ---------------------------------------------------------------------------

def _invariant_running_not_deleted(sm: StateMachine) -> List[str]:
    """A running job must not be marked as removed."""
    if sm.state == "running" and sm.context.get("removed"):
        return ["RUNNING_DELETED: running job is marked as removed"]
    return []


def _invariant_paused_never_fires(sm: StateMachine) -> List[str]:
    """A paused job must not have a pending tick."""
    if sm.state == "paused" and sm.context.get("tick_pending"):
        return ["PAUSED_TICK: paused job has tick_pending=True"]
    return []


def _invariant_repeat_consistency(sm: StateMachine) -> List[str]:
    """runs_completed + runs_left == total_repeat (when defined)."""
    total = sm.context.get("repeat_total")
    if total is None:
        return []

    completed = sm.context.get("runs_completed", 0)
    left = sm.context.get("runs_left", total)

    if completed + left != total:
        return [
            f"REPEAT_MISMATCH: completed({completed}) + left({left}) != total({total})"
        ]
    return []


def _invariant_completed_no_next_run(sm: StateMachine) -> List[str]:
    """Completed jobs should not have a next_run_at."""
    if sm.state == "completed" and sm.context.get("next_run_at") is not None:
        return [
            f"COMPLETED_WITH_NEXT_RUN: completed job has "
            f"next_run_at={sm.context.get('next_run_at')}"
        ]
    return []


def _invariant_scheduled_has_next_run(sm: StateMachine) -> List[str]:
    """Scheduled jobs should have a next_run_at."""
    if sm.state == "scheduled" and sm.context.get("next_run_at") is None:
        # Only warn if the job was fully initialized
        if sm.context.get("initialized"):
            return ["SCHEDULED_NO_NEXT_RUN: scheduled job has no next_run_at"]
    return []


def _invariant_valid_state(sm: StateMachine) -> List[str]:
    """Current state must be a valid cron state."""
    if sm.state not in CRON_STATES:
        return [f"INVALID_STATE: '{sm.state}' is not a valid cron job state"]
    return []


ALL_CRON_INVARIANTS = [
    _invariant_valid_state,
    _invariant_running_not_deleted,
    _invariant_paused_never_fires,
    _invariant_repeat_consistency,
    _invariant_completed_no_next_run,
    _invariant_scheduled_has_next_run,
]


# ---------------------------------------------------------------------------
# CronJobStateMachine
# ---------------------------------------------------------------------------

class CronJobStateMachine:
    """State machine for a single cron job lifecycle.

    Wraps the generic StateMachine with cron-specific transitions, guards,
    and invariants matching SM-1 from the improvement plan.

    The context dict stores job-associated data:
        - job_id: str
        - recurring: bool
        - repeat_total: Optional[int]
        - runs_completed: int
        - runs_left: Optional[int]
        - next_run_at: Optional[datetime]
        - retry_count: int
        - max_retries: int
        - enabled: bool
        - initialized: bool
        - last_run_at: Optional[datetime]
        - last_error: Optional[str]
    """

    def __init__(
        self,
        job_id: str,
        initial_state: str = "nonexistent",
        check_invariants: bool = True,
        on_transition: Optional[Callable[[TransitionRecord, StateMachine], None]] = None,
    ):
        self.job_id = job_id

        transitions = {
            "create": [
                TransitionDef("nonexistent", "scheduled"),
            ],
            "tick": [
                TransitionDef("scheduled", "running", _guard_tick, "enabled_and_due"),
            ],
            "mark_success": [
                TransitionDef(
                    "running", "scheduled",
                    _guard_mark_success_recurring,
                    "recurring_with_runs_left",
                ),
                TransitionDef(
                    "running", "completed",
                    _guard_mark_success_completed,
                    "one_shot_or_exhausted",
                ),
            ],
            "mark_failure": [
                TransitionDef("running", "failed"),
            ],
            "retry": [
                TransitionDef("failed", "scheduled", _guard_retry, "retry_count_ok"),
            ],
            "pause": [
                TransitionDef("scheduled", "paused"),
                TransitionDef("failed", "paused"),
            ],
            "resume": [
                TransitionDef("paused", "scheduled"),
            ],
            "trigger": [
                TransitionDef("paused", "scheduled"),
            ],
            "remove": [
                TransitionDef("scheduled", "removed", _guard_remove, "not_running"),
                TransitionDef("paused", "removed", _guard_remove, "not_running"),
                TransitionDef("completed", "removed", _guard_remove, "not_running"),
                TransitionDef("failed", "removed", _guard_remove, "not_running"),
                TransitionDef("nonexistent", "removed", _guard_remove, "not_running"),
            ],
            "recover": [
                TransitionDef("running", "failed", _guard_recover, "stale_lock"),
            ],
        }

        self._sm = StateMachine(
            name=f"cron:{job_id}",
            states=CRON_STATES,
            initial_state=initial_state,
            transitions=transitions,
            invariants=list(ALL_CRON_INVARIANTS),
            on_transition=on_transition,
            check_invariants_on_transition=check_invariants,
        )

        # Initialize context
        self._sm.context.update({
            "job_id": job_id,
            "recurring": False,
            "repeat_total": None,
            "runs_completed": 0,
            "runs_left": None,
            "next_run_at": None,
            "retry_count": 0,
            "max_retries": 3,
            "enabled": True,
            "initialized": False,
            "last_run_at": None,
            "last_error": None,
            "tick_pending": False,
            "removed": False,
        })

    # ----- Properties -----

    @property
    def state(self) -> str:
        return self._sm.state

    @property
    def context(self) -> Dict[str, Any]:
        return self._sm.context

    @property
    def machine(self) -> StateMachine:
        """Access the underlying StateMachine directly."""
        return self._sm

    @property
    def history(self) -> List[TransitionRecord]:
        return self._sm.history

    # ----- High-level operations -----

    def create(
        self,
        next_run_at: Optional[datetime] = None,
        recurring: bool = False,
        repeat_total: Optional[int] = None,
        max_retries: int = 3,
        enabled: bool = True,
    ) -> TransitionRecord:
        """Create/schedule the job: nonexistent -> scheduled."""
        self._sm.context.update({
            "next_run_at": next_run_at,
            "recurring": recurring,
            "repeat_total": repeat_total,
            "runs_left": repeat_total,
            "max_retries": max_retries,
            "enabled": enabled,
            "initialized": True,
        })
        return self._sm.apply("create")

    def tick(self, now: Optional[datetime] = None) -> TransitionRecord:
        """Attempt to run the job: scheduled -> running."""
        ctx = {
            "enabled": self._sm.context.get("enabled", True),
            "now": now or datetime.now(),
            "next_run_at": self._sm.context.get("next_run_at"),
        }
        record = self._sm.apply("tick", context=ctx)
        self._sm.context["last_run_at"] = ctx["now"]
        self._sm.context["tick_pending"] = False
        return record

    def mark_success(
        self,
        next_run_at: Optional[datetime] = None,
    ) -> TransitionRecord:
        """Mark job run as successful: running -> scheduled or completed.

        The abstract TLA+ action ``MarkSuccessDone`` atomically clears
        ``nextRunDue`` in the same step as the state transition.  To refine
        that faithfully the context here is pre-updated for whichever branch
        the guards will pick, so ``TerminalNoNextRun`` holds during the
        invariant check that ``apply`` runs.
        """
        recurring = self._sm.context.get("recurring", False)
        runs_left = self._sm.context.get("runs_left")
        will_complete = (not recurring) or (
            runs_left is not None and runs_left <= 1
        )

        # Atomically stage the target next_run_at before the transition
        # so the invariant check sees a consistent context.
        if will_complete:
            self._sm.context["next_run_at"] = None
            self._sm.context["enabled"] = False
        else:
            self._sm.context["next_run_at"] = next_run_at

        ctx = {
            "recurring": recurring,
            "runs_left": runs_left,
        }
        record = self._sm.apply("mark_success", context=ctx)

        # Post-apply: update counters (these don't participate in the
        # atomic guard/invariant set above).
        self._sm.context["runs_completed"] = (
            self._sm.context.get("runs_completed", 0) + 1
        )
        if self._sm.context.get("runs_left") is not None:
            self._sm.context["runs_left"] = max(
                0, self._sm.context["runs_left"] - 1
            )
        self._sm.context["retry_count"] = 0
        self._sm.context["last_error"] = None

        return record

    def mark_failure(self, error: Optional[str] = None) -> TransitionRecord:
        """Mark job run as failed: running -> failed."""
        record = self._sm.apply("mark_failure")
        self._sm.context["last_error"] = error
        self._sm.context["retry_count"] = (
            self._sm.context.get("retry_count", 0) + 1
        )
        return record

    def retry(
        self,
        next_run_at: Optional[datetime] = None,
    ) -> TransitionRecord:
        """Retry a failed job: failed -> scheduled."""
        ctx = {
            "retry_count": self._sm.context.get("retry_count", 0),
            "max_retries": self._sm.context.get("max_retries", 3),
        }
        record = self._sm.apply("retry", context=ctx)
        self._sm.context["next_run_at"] = next_run_at
        return record

    def pause(self) -> TransitionRecord:
        """Pause the job: scheduled/failed -> paused."""
        record = self._sm.apply("pause")
        self._sm.context["enabled"] = False
        self._sm.context["tick_pending"] = False
        return record

    def resume(
        self,
        next_run_at: Optional[datetime] = None,
    ) -> TransitionRecord:
        """Resume a paused job: paused -> scheduled."""
        record = self._sm.apply("resume")
        self._sm.context["enabled"] = True
        if next_run_at:
            self._sm.context["next_run_at"] = next_run_at
        return record

    def trigger(self) -> TransitionRecord:
        """Force-trigger a paused job: paused -> scheduled (immediate)."""
        record = self._sm.apply("trigger")
        self._sm.context["enabled"] = True
        return record

    def remove(self) -> TransitionRecord:
        """Remove the job: any (except running) -> removed."""
        ctx = {}  # guard reads current state directly
        record = self._sm.apply("remove", context=ctx)
        self._sm.context["removed"] = True
        self._sm.context["enabled"] = False
        self._sm.context["next_run_at"] = None
        return record

    def recover(self) -> TransitionRecord:
        """Recover a stale running job: running -> failed.

        Should be called when a stale lock is detected (e.g., process crash).
        """
        ctx = {"stale_lock": True}
        record = self._sm.apply("recover", context=ctx)
        self._sm.context["last_error"] = "Recovered from stale lock"
        return record

    # ----- Recovery helpers -----

    def auto_retry_if_possible(
        self,
        next_run_at: Optional[datetime] = None,
    ) -> Optional[TransitionRecord]:
        """Attempt auto-retry if in failed state and retries remain.

        Returns the transition record if retry succeeded, None otherwise.
        """
        if self._sm.state != "failed":
            return None

        retry_count = self._sm.context.get("retry_count", 0)
        max_retries = self._sm.context.get("max_retries", 3)

        if retry_count >= max_retries:
            logger.info(
                "[cron:%s] Auto-retry exhausted (%d/%d)",
                self.job_id,
                retry_count,
                max_retries,
            )
            return None

        try:
            return self.retry(next_run_at=next_run_at)
        except (InvalidTransition, GuardFailed) as e:
            logger.warning("[cron:%s] Auto-retry failed: %s", self.job_id, e)
            return None

    def recover_if_stale(
        self,
        max_running_seconds: float = 3600,
    ) -> Optional[TransitionRecord]:
        """Check if a running job is stale and recover it.

        A job is considered stale if it has been running longer than
        max_running_seconds.
        """
        if self._sm.state != "running":
            return None

        last_run = self._sm.context.get("last_run_at")
        if last_run is None:
            return None

        if isinstance(last_run, datetime):
            elapsed = (datetime.now() - last_run).total_seconds()
        else:
            elapsed = time.time() - float(last_run)

        if elapsed > max_running_seconds:
            logger.warning(
                "[cron:%s] Stale job detected (running for %.0fs > %.0fs)",
                self.job_id,
                elapsed,
                max_running_seconds,
            )
            return self.recover()

        return None

    # ----- Invariant checking -----

    def check_invariants(self) -> List[str]:
        """Run all cron-specific invariants."""
        return self._sm.check_invariants()

    # ----- Utilities -----

    def get_available_actions(self) -> List[str]:
        """Actions available from current state."""
        return self._sm.get_available_actions()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for debugging."""
        d = self._sm.to_dict()
        d["job_id"] = self.job_id
        return d

    def __repr__(self) -> str:
        return (
            f"CronJobStateMachine(job={self.job_id}, state={self.state}, "
            f"actions={self.get_available_actions()})"
        )


# ---------------------------------------------------------------------------
# Bulk operations (for checking all jobs at once)
# ---------------------------------------------------------------------------

def check_all_cron_invariants(
    machines: List[CronJobStateMachine],
) -> List[str]:
    """Check invariants across all cron job state machines.

    Additional cross-job invariants:
    - No duplicate job IDs
    - At most one job running at a time (if tick lock is required)
    """
    violations = []

    # Per-job invariants
    for sm in machines:
        job_violations = sm.check_invariants()
        for v in job_violations:
            violations.append(f"[{sm.job_id}] {v}")

    # Cross-job: duplicate IDs
    seen_ids = set()
    for sm in machines:
        if sm.job_id in seen_ids:
            violations.append(f"DUPLICATE_JOB_ID: '{sm.job_id}'")
        seen_ids.add(sm.job_id)

    # Cross-job: concurrent running count
    running = [sm for sm in machines if sm.state == "running"]
    if len(running) > 1:
        ids = [sm.job_id for sm in running]
        violations.append(
            f"CONCURRENT_RUNNING: {len(running)} jobs running simultaneously: {ids}"
        )

    return violations
