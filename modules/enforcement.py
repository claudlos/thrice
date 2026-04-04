"""
Enforcement mode configuration for Hermes runtime invariant checking.

Controls how invariant violations are handled across different environments.
Set via HERMES_INVARIANT_MODE environment variable.

Modes:
    DEVELOPMENT  -> raise InvariantViolation immediately (fail-fast)
    PRODUCTION   -> log warning, collect violations, don't crash
    TESTING      -> collect all violations, assert at end of test

Usage:
    from enforcement import get_enforcement_mode, enforce, EnforcementMode

    mode = get_enforcement_mode()
    enforce(violations, context="cron", mode=mode)
"""

import enum
import logging
import os
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class InvariantViolation(Exception):
    """Raised when a critical invariant is broken in DEVELOPMENT mode."""

    def __init__(self, message: str, violations: Optional[List[str]] = None):
        super().__init__(message)
        self.violations = violations or [message]


class EnforcementMode(enum.Enum):
    """How invariant violations are handled."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


# ---------------------------------------------------------------------------
# Mapping from env var strings to EnforcementMode
# ---------------------------------------------------------------------------

_ENV_TO_MODE = {
    # DEVELOPMENT aliases
    "development": EnforcementMode.DEVELOPMENT,
    "dev": EnforcementMode.DEVELOPMENT,
    "strict": EnforcementMode.DEVELOPMENT,
    # PRODUCTION aliases
    "production": EnforcementMode.PRODUCTION,
    "prod": EnforcementMode.PRODUCTION,
    "warn": EnforcementMode.PRODUCTION,
    # TESTING aliases
    "testing": EnforcementMode.TESTING,
    "test": EnforcementMode.TESTING,
    "collect": EnforcementMode.TESTING,
    # Silent is mapped to production with a flag
    "silent": EnforcementMode.PRODUCTION,
}


def get_enforcement_mode() -> EnforcementMode:
    """Read enforcement mode from HERMES_INVARIANT_MODE env var.

    Returns EnforcementMode.PRODUCTION if unset or unrecognized.
    """
    raw = os.getenv("HERMES_INVARIANT_MODE", "warn").lower().strip()
    mode = _ENV_TO_MODE.get(raw)
    if mode is None:
        logger.warning(
            "Unknown HERMES_INVARIANT_MODE=%r, defaulting to PRODUCTION", raw
        )
        return EnforcementMode.PRODUCTION
    return mode


def is_silent() -> bool:
    """Check if mode is explicitly set to 'silent' (suppresses warnings too)."""
    raw = os.getenv("HERMES_INVARIANT_MODE", "warn").lower().strip()
    return raw == "silent"


# ---------------------------------------------------------------------------
# Violation collector for TESTING mode
# ---------------------------------------------------------------------------

class ViolationCollector:
    """Collects violations during test runs for later assertion.

    Usage in tests:
        collector = ViolationCollector()
        enforce(violations, context="cron", collector=collector)
        # ... more operations ...
        collector.assert_clean()  # raises if any violations were collected
    """

    def __init__(self):
        self._violations: List[str] = []

    def add(self, violations: List[str]) -> None:
        self._violations.extend(violations)

    @property
    def violations(self) -> List[str]:
        return list(self._violations)

    @property
    def count(self) -> int:
        return len(self._violations)

    def clear(self) -> None:
        self._violations.clear()

    def assert_clean(self, message: str = "") -> None:
        """Assert no violations have been collected."""
        if self._violations:
            detail = "\n  ".join(self._violations)
            prefix = f"{message}: " if message else ""
            raise AssertionError(
                f"{prefix}{len(self._violations)} invariant violation(s):\n  {detail}"
            )

    def __repr__(self) -> str:
        return f"ViolationCollector(count={self.count})"


# Global collector for TESTING mode (reset between tests)
_global_collector = ViolationCollector()


def get_global_collector() -> ViolationCollector:
    """Get the global violation collector (for TESTING mode)."""
    return _global_collector


def reset_global_collector() -> None:
    """Reset the global collector (call in test setUp/tearDown)."""
    _global_collector.clear()


# ---------------------------------------------------------------------------
# Core enforcement function
# ---------------------------------------------------------------------------

def enforce(
    violations: List[str],
    context: str = "",
    mode: Optional[EnforcementMode] = None,
    collector: Optional[ViolationCollector] = None,
    on_violation: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Handle violations according to the enforcement mode.

    Args:
        violations: List of violation description strings.
        context: Subsystem name for log messages (e.g., "cron", "messages").
        mode: Override enforcement mode (defaults to env var).
        collector: Optional ViolationCollector (used in TESTING mode).
        on_violation: Optional callback for each violation string.

    Returns:
        The violations list (unchanged), for chaining.

    Raises:
        InvariantViolation: In DEVELOPMENT mode, on the first violation.
    """
    if not violations:
        return violations

    if mode is None:
        mode = get_enforcement_mode()

    prefix = f"[{context}] " if context else ""

    # Optional callback
    if on_violation:
        for v in violations:
            on_violation(f"{prefix}{v}")

    if mode == EnforcementMode.DEVELOPMENT:
        # Fail-fast: raise on the first violation
        msg = f"INVARIANT VIOLATION: {prefix}{violations[0]}"
        if len(violations) > 1:
            msg += f" (and {len(violations) - 1} more)"
        raise InvariantViolation(msg, violations)

    elif mode == EnforcementMode.PRODUCTION:
        if not is_silent():
            for v in violations:
                logger.warning("INVARIANT: %s%s", prefix, v)

    elif mode == EnforcementMode.TESTING:
        target = collector or _global_collector
        formatted = [f"{prefix}{v}" for v in violations]
        target.add(formatted)
        # Also log at debug level for test output visibility
        for v in violations:
            logger.debug("INVARIANT (collected): %s%s", prefix, v)

    return violations


# ---------------------------------------------------------------------------
# Decorator for invariant-checked functions
# ---------------------------------------------------------------------------

def check_invariants_after(
    checker_fn: Callable[..., List[str]],
    context: str = "",
    mode: Optional[EnforcementMode] = None,
):
    """Decorator that runs an invariant checker after the decorated function.

    The checker_fn receives the return value of the decorated function.

    Example:
        @check_invariants_after(
            lambda result: InvariantChecker.check_cron_invariants(result),
            context="cron"
        )
        def update_job(job_id, new_state):
            ...
            return all_jobs
    """
    def decorator(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            violations = checker_fn(result)
            enforce(violations, context=context, mode=mode)
            return result

        return wrapper
    return decorator
