"""
Pytest configuration for Hermes improvement tests.

Provides:
  - Custom markers for test levels (property, invariant, refinement, slow)
  - Hypothesis profiles (ci, dev, exhaustive)
  - Shared fixtures for common test data factories
  - sys.path setup so thrice/ invariants are importable
"""

import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Path setup: make thrice/ importable from tests/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THRICE_ROOT = os.path.join(_PROJECT_ROOT, "thrice")
_NEW_FILES = os.path.join(_PROJECT_ROOT, "new-files")

for p in (_PROJECT_ROOT, _THRICE_ROOT, _NEW_FILES):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck, Phase, Verbosity, settings

    # CI profile: fast, fewer examples, skip shrinking
    settings.register_profile(
        "ci",
        max_examples=30,
        stateful_step_count=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
        phases=[Phase.explicit, Phase.generate],
        derandomize=True,
    )

    # Dev profile: moderate exploration
    settings.register_profile(
        "dev",
        max_examples=100,
        stateful_step_count=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )

    # Exhaustive profile: thorough exploration for nightly runs
    settings.register_profile(
        "exhaustive",
        max_examples=500,
        stateful_step_count=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
        verbosity=Verbosity.verbose,
    )

    # Load profile from HYPOTHESIS_PROFILE env var, default to "ci"
    _profile = os.getenv("HYPOTHESIS_PROFILE", "ci")
    settings.load_profile(_profile)

except ImportError:
    pass  # hypothesis not installed; property tests will be skipped


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "property: property-based test (hypothesis)")
    config.addinivalue_line("markers", "invariant: invariant checker unit test")
    config.addinivalue_line("markers", "refinement: abstract/concrete refinement test")
    config.addinivalue_line("markers", "slow: slow test (>5s)")
    config.addinivalue_line("markers", "stateful: hypothesis stateful/RuleBasedStateMachine test")
    config.addinivalue_line(
        "markers",
        "requires_hermes: test needs a live hermes-agent checkout on PYTHONPATH",
    )


def _hermes_available() -> bool:
    """Return True iff the live hermes-agent package is importable.

    Checks for the real cron.jobs symbols, not the test stub below, so
    a failed import or a stub-only installation both count as "not
    available".
    """
    import importlib
    try:
        mod = importlib.import_module("cron.jobs")
    except Exception:
        return False
    # The stub has JOB_STATES but not the full API (e.g. create_job, get_job).
    return all(hasattr(mod, name) for name in ("create_job", "get_job"))


_HERMES_PRESENT = _hermes_available()


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``requires_hermes`` when Hermes isn't on PYTHONPATH."""
    if _HERMES_PRESENT:
        return
    skip_no_hermes = pytest.mark.skip(
        reason="hermes-agent not on PYTHONPATH; skipping requires_hermes tests"
    )
    for item in items:
        if "requires_hermes" in item.keywords:
            item.add_marker(skip_no_hermes)


# ---------------------------------------------------------------------------
# Stub modules for imports that depend on Hermes runtime
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_cron_importable():
    """Ensure cron.jobs is importable with JOB_STATES for invariant tests."""
    already = "cron.jobs" in sys.modules
    if not already:
        if "cron" not in sys.modules:
            cron_stub = types.ModuleType("cron")
            cron_stub.__path__ = []
            sys.modules["cron"] = cron_stub
        stub = types.ModuleType("cron.jobs")
        stub.JOB_STATES = frozenset(
            {"scheduled", "running", "paused", "completed", "failed"}
        )
        sys.modules["cron.jobs"] = stub
    yield
    if not already and "cron.jobs" in sys.modules:
        del sys.modules["cron.jobs"]


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

@pytest.fixture
def make_message():
    """Factory for chat messages."""
    def _make(role="user", content="hello", tool_calls=None, tool_call_id=None):
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": cid, "function": {"name": "test", "arguments": "{}"}}
                for cid in tool_calls
            ]
            msg["content"] = None
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg
    return _make


@pytest.fixture
def make_process_session():
    """Factory for process session dicts."""
    import time
    def _make(sid="proc_1", pid=12345, exited=False, exit_code=None,
              started_at=None, detached=False, command="echo test"):
        return {
            "id": sid,
            "command": command,
            "pid": pid,
            "exited": exited,
            "exit_code": exit_code,
            "started_at": started_at or time.time(),
            "detached": detached,
        }
    return _make


@pytest.fixture
def make_cron_job():
    """Factory for cron job dicts."""
    from datetime import datetime, timedelta, timezone

    def _make(jid="job_1", name="test", enabled=True, schedule_kind="interval",
              repeat_times=None, repeat_completed=0, next_run_at="auto",
              last_status=None, last_error=None, last_run_at=None, created_at=None):
        now = datetime.now(timezone.utc)
        if next_run_at == "auto":
            next_run_at = (now + timedelta(hours=1)).isoformat() if enabled else None
        return {
            "id": jid,
            "name": name,
            "prompt": "do something",
            "schedule": {"kind": schedule_kind, "minutes": 30, "display": "every 30m"},
            "repeat": {"times": repeat_times, "completed": repeat_completed},
            "enabled": enabled,
            "created_at": created_at or (now - timedelta(hours=2)).isoformat(),
            "next_run_at": next_run_at,
            "last_run_at": last_run_at,
            "last_status": last_status,
            "last_error": last_error,
        }
    return _make
