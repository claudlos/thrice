"""
Unit tests for every cron invariant from SM-1 specification.

Tests the thrice/invariants/cron_invariants.py checker against all
invariant formulas defined in specs/SM-1-cron-lifecycle.md:

  INV-CJ1  RepeatCountConsistency
  INV-CJ2  CompletedNonNegative
  INV-CJ3  NextRunConsistency
  INV-CJ4  LastRunBeforeNextRun
  INV-CJ5  CreatedAtPrecedes
  INV-CJ6  UniqueJobIds
  INV-CJ7  ValidScheduleKind
  INV-CJ8  StatusConsistency
  INV-CJ9  CompletedMonotonic

Plus state transition validation and state inference.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "thrice"))

# ``invariants.cron_invariants`` lives inside the hermes-agent checkout.
# Skip collection if it's missing so module-level imports don't error out.
pytest.importorskip(
    "invariants.cron_invariants",
    reason="hermes-agent invariants.cron_invariants not available",
)

from invariants.cron_invariants import (  # noqa: E402
    _VALID_CRON_TRANSITIONS,
    CRON_STATES,
    VALID_SCHEDULE_KINDS,
    CronJobInvariantChecker,
    infer_cron_state,
    validate_jobs_list,
)

pytestmark = [pytest.mark.invariant, pytest.mark.requires_hermes]


# -- Helpers ------------------------------------------------------------------

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _past_iso(minutes=60):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

def _future_iso(minutes=60):
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()

def make_job(jid="job_1", enabled=True, schedule_kind="interval",
             repeat_times=None, repeat_completed=0,
             next_run_at="auto", last_run_at=None,
             last_status=None, last_error=None, created_at=None):
    if next_run_at == "auto":
        next_run_at = _future_iso() if enabled else None
    return {
        "id": jid, "name": "test", "prompt": "do something",
        "schedule": {"kind": schedule_kind, "minutes": 30, "display": "every 30m"},
        "repeat": {"times": repeat_times, "completed": repeat_completed},
        "enabled": enabled, "created_at": created_at or _past_iso(120),
        "next_run_at": next_run_at, "last_run_at": last_run_at,
        "last_status": last_status, "last_error": last_error,
    }


# ============================================================================
# INV-CJ6: UniqueJobIds
# ============================================================================

class TestUniqueJobIds:
    def test_unique_passes(self):
        assert CronJobInvariantChecker.check_unique_job_ids(
            [make_job("j1"), make_job("j2")]
        ) is None

    def test_duplicate_caught(self):
        err = CronJobInvariantChecker.check_unique_job_ids(
            [make_job("j1"), make_job("j1")]
        )
        assert err is not None
        assert err.invariant_name == "UniqueJobIds"

    def test_empty_passes(self):
        assert CronJobInvariantChecker.check_unique_job_ids([]) is None


# ============================================================================
# INV-CJ7: ValidScheduleKind
# ============================================================================

class TestValidScheduleKind:
    @pytest.mark.parametrize("kind", sorted(VALID_SCHEDULE_KINDS))
    def test_valid_kinds(self, kind):
        assert CronJobInvariantChecker.check_valid_schedule_kind(
            [make_job(schedule_kind=kind)]
        ) is None

    def test_invalid_kind(self):
        err = CronJobInvariantChecker.check_valid_schedule_kind(
            [make_job(schedule_kind="weekly")]
        )
        assert err is not None
        assert err.invariant_name == "ValidScheduleKind"

    def test_non_dict_schedule(self):
        job = make_job()
        job["schedule"] = "not_a_dict"
        err = CronJobInvariantChecker.check_valid_schedule_kind([job])
        assert err is not None


# ============================================================================
# INV-CJ1: RepeatCountConsistency
# ============================================================================

class TestRepeatCountConsistency:
    def test_under_limit(self):
        assert CronJobInvariantChecker.check_repeat_count_consistency(
            [make_job(repeat_times=5, repeat_completed=3)]
        ) is None

    def test_at_limit(self):
        assert CronJobInvariantChecker.check_repeat_count_consistency(
            [make_job(repeat_times=5, repeat_completed=5)]
        ) is None

    def test_over_limit(self):
        err = CronJobInvariantChecker.check_repeat_count_consistency(
            [make_job(repeat_times=3, repeat_completed=5)]
        )
        assert err is not None
        assert err.invariant_name == "RepeatCountConsistency"

    def test_unlimited(self):
        assert CronJobInvariantChecker.check_repeat_count_consistency(
            [make_job(repeat_times=None, repeat_completed=100)]
        ) is None


# ============================================================================
# INV-CJ2: CompletedNonNegative
# ============================================================================

class TestCompletedNonNegative:
    def test_zero(self):
        assert CronJobInvariantChecker.check_completed_non_negative(
            [make_job(repeat_completed=0)]
        ) is None

    def test_positive(self):
        assert CronJobInvariantChecker.check_completed_non_negative(
            [make_job(repeat_completed=10)]
        ) is None

    def test_negative(self):
        err = CronJobInvariantChecker.check_completed_non_negative(
            [make_job(repeat_completed=-1)]
        )
        assert err is not None
        assert err.invariant_name == "CompletedNonNegative"


# ============================================================================
# INV-CJ8: StatusConsistency
# ============================================================================

class TestStatusConsistency:
    def test_none_status_no_error(self):
        assert CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status=None, last_error=None)]
        ) is None

    def test_ok_no_error(self):
        assert CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status="ok", last_error=None)]
        ) is None

    def test_error_with_message(self):
        assert CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status="error", last_error="timeout")]
        ) is None

    def test_invalid_status(self):
        err = CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status="pending")]
        )
        assert err is not None

    def test_error_without_message(self):
        err = CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status="error", last_error=None)]
        )
        assert err is not None

    def test_ok_with_stale_error(self):
        err = CronJobInvariantChecker.check_status_consistency(
            [make_job(last_status="ok", last_error="old")]
        )
        assert err is not None


# ============================================================================
# INV-CJ3: NextRunConsistency
# ============================================================================

class TestNextRunConsistency:
    def test_enabled_with_next_run(self):
        assert CronJobInvariantChecker.check_next_run_consistency(
            [make_job(enabled=True)]
        ) is None

    def test_enabled_no_next_run_never_ran(self):
        err = CronJobInvariantChecker.check_next_run_consistency(
            [make_job(enabled=True, next_run_at=None, last_status=None)]
        )
        assert err is not None
        assert err.invariant_name == "NextRunConsistency"

    def test_enabled_no_next_run_already_ran(self):
        assert CronJobInvariantChecker.check_next_run_consistency(
            [make_job(enabled=True, next_run_at=None, last_status="ok")]
        ) is None


# ============================================================================
# INV-CJ4: LastRunBeforeNextRun
# ============================================================================

class TestLastRunBeforeNextRun:
    def test_correct_order(self):
        job = make_job(last_run_at=_past_iso(30), next_run_at=_future_iso(30))
        assert CronJobInvariantChecker.check_last_run_before_next_run(job) is None

    def test_reversed_order(self):
        job = make_job(last_run_at=_future_iso(30), next_run_at=_past_iso(30))
        err = CronJobInvariantChecker.check_last_run_before_next_run(job)
        assert err is not None
        assert err.invariant_name == "LastRunBeforeNextRun"

    def test_neither_set(self):
        job = make_job(last_run_at=None, next_run_at=None, enabled=False)
        assert CronJobInvariantChecker.check_last_run_before_next_run(job) is None


# ============================================================================
# INV-CJ5: CreatedAtPrecedesLastRun
# ============================================================================

class TestCreatedAtPrecedes:
    def test_correct_order(self):
        job = make_job(created_at=_past_iso(120), last_run_at=_past_iso(30))
        assert CronJobInvariantChecker.check_created_at_precedes_last_run(job) is None

    def test_created_after_last_run(self):
        job = make_job(created_at=_future_iso(30), last_run_at=_past_iso(30))
        err = CronJobInvariantChecker.check_created_at_precedes_last_run(job)
        assert err is not None
        assert err.invariant_name == "CreatedAtPrecedesLastRun"


# ============================================================================
# INV-CJ9: CompletedMonotonic
# ============================================================================

class TestCompletedMonotonic:
    def test_increasing(self):
        old = make_job(repeat_completed=3)
        new = make_job(repeat_completed=5)
        assert CronJobInvariantChecker.check_completed_monotonic(old, new) is None

    def test_equal(self):
        old = make_job(repeat_completed=3)
        new = make_job(repeat_completed=3)
        assert CronJobInvariantChecker.check_completed_monotonic(old, new) is None

    def test_decreasing(self):
        old = make_job(repeat_completed=5)
        new = make_job(repeat_completed=3)
        err = CronJobInvariantChecker.check_completed_monotonic(old, new)
        assert err is not None
        assert err.invariant_name == "CompletedMonotonic"


# ============================================================================
# State transitions
# ============================================================================

class TestValidTransitions:
    def test_all_valid_transitions_accepted(self):
        for source, targets in _VALID_CRON_TRANSITIONS.items():
            for target in targets:
                assert CronJobInvariantChecker.check_valid_transition(source, target) is None, (
                    f"{source} -> {target} should be valid"
                )

    def test_deleted_is_terminal(self):
        for target in CRON_STATES - {"deleted"}:
            err = CronJobInvariantChecker.check_valid_transition("deleted", target)
            assert err is not None, f"deleted -> {target} should be rejected"

    def test_self_transitions_are_noop(self):
        for state in CRON_STATES:
            assert CronJobInvariantChecker.check_valid_transition(state, state) is None


# ============================================================================
# State inference
# ============================================================================

class TestInferCronState:
    def test_running(self):
        assert infer_cron_state(make_job(), is_running=True) == "running"

    def test_disabled(self):
        assert infer_cron_state(make_job(enabled=False)) == "disabled"

    def test_scheduled_never_ran(self):
        assert infer_cron_state(make_job(last_status=None)) == "scheduled"

    def test_completed_no_next_run(self):
        job = make_job(last_status="ok", last_run_at=_past_iso(), next_run_at=None)
        assert infer_cron_state(job) == "completed"

    def test_failed_no_next_run(self):
        job = make_job(last_status="error", last_error="x",
                       last_run_at=_past_iso(), next_run_at=None)
        assert infer_cron_state(job) == "failed"


# ============================================================================
# validate_jobs_list integration
# ============================================================================

class TestValidateJobsList:
    def test_valid_list(self):
        jobs = [make_job("j1"), make_job("j2")]
        errors = validate_jobs_list(jobs)
        assert errors == []

    def test_multiple_violations(self):
        jobs = [
            make_job("j1", repeat_completed=-1),
            make_job("j1"),  # duplicate
        ]
        errors = validate_jobs_list(jobs)
        assert len(errors) >= 1

    def test_empty_list(self):
        errors = validate_jobs_list([])
        assert errors == []


# ============================================================================
# Liveness warnings
# ============================================================================

class TestLivenessWarnings:
    def test_overdue_job_warned(self):
        now = datetime.now(timezone.utc)
        past = (now - timedelta(minutes=30)).isoformat()
        warnings = CronJobInvariantChecker.warn_overdue_jobs(
            [make_job(enabled=True, next_run_at=past)],
            now_iso=now.isoformat()
        )
        assert len(warnings) == 1

    def test_failed_job_warned(self):
        warnings = CronJobInvariantChecker.warn_high_failure_rate(
            [make_job(last_status="error", last_error="timeout")]
        )
        assert len(warnings) == 1

    def test_disabled_job_warned(self):
        warnings = CronJobInvariantChecker.warn_stale_disabled_jobs(
            [make_job(enabled=False)]
        )
        assert len(warnings) == 1
