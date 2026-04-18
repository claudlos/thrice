"""Stack + local-variable trace capture for Hermes Agent (Thrice).

When a test (or any callable) fails, ``trace_capture`` grabs a compact
"what was this function actually *doing*" record - the failing frame's
local variables plus a short traceback - so the agent's next turn gets
the execution context without re-running anything.

Design choices:

- **Safe-by-default**: a value is serialised only if ``repr(v)`` is short
  and doesn't raise.  Big / opaque / infinite-recursion reprs are elided.
- **PII-aware**: keys matching ``password|token|secret|key|auth`` are
  redacted.  This layer pairs with ``secret_scanner``.
- **Zero test-runner coupling**: the public entry points take either a
  callable or an exception+traceback.  Works with pytest, unittest, or a
  plain ``try/except``.

Typical usage::

    from trace_capture import capture

    record = capture(lambda: run_once())
    if not record.ok:
        print(record.format_agent_context())
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from types import FrameType, TracebackType
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameRecord:
    """One frame from the captured traceback."""

    file: str
    line: int
    function: str
    locals: Dict[str, str]    # name -> safe-repr string


@dataclass
class TraceRecord:
    """The result of ``capture(...)``."""

    ok: bool
    exc_type: Optional[str] = None
    exc_message: Optional[str] = None
    frames: List[FrameRecord] = field(default_factory=list)
    duration_s: float = 0.0
    value: Any = None          # the return value when ok=True

    @property
    def bottom_frame(self) -> Optional[FrameRecord]:
        return self.frames[-1] if self.frames else None

    def format_agent_context(self, max_frames: int = 3, max_locals: int = 8) -> str:
        """Compact human/agent-readable block suitable for prompt injection."""
        if self.ok:
            return "trace_capture: OK"

        lines = [
            f"{self.exc_type or 'Exception'}: {self.exc_message or ''}".rstrip(),
            "",
            "=== stack (most recent call last) ===",
        ]
        for frame in self.frames[-max_frames:]:
            lines.append(f"{frame.file}:{frame.line} in {frame.function}()")
            for i, (k, v) in enumerate(frame.locals.items()):
                if i >= max_locals:
                    lines.append(f"  ... ({len(frame.locals) - max_locals} more locals)")
                    break
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Safe repr
# ---------------------------------------------------------------------------

_MAX_REPR = 160
_SENSITIVE_NAME_RE = re.compile(
    r"(?i)password|passwd|pwd|secret|token|apikey|api_key|credential|auth"
)


def _safe_repr(value: Any) -> str:
    """Return a bounded, single-line ``repr`` of ``value``; never raises."""
    try:
        r = repr(value)
    except Exception as exc:   # __repr__ itself blew up
        return f"<repr-error: {type(exc).__name__}>"
    r = r.replace("\n", "\\n")
    if len(r) > _MAX_REPR:
        r = r[: _MAX_REPR - 3] + "..."
    return r


def _safe_locals(frame: FrameType, max_vars: int = 32) -> Dict[str, str]:
    """Return a safe, redacted, bounded dict of the frame's locals."""
    out: Dict[str, str] = {}
    for i, (name, value) in enumerate(frame.f_locals.items()):
        if i >= max_vars:
            out["...__truncated__"] = f"<{len(frame.f_locals) - max_vars} more locals>"
            break
        if name.startswith("__") and name.endswith("__"):
            # skip dunder noise (__builtins__, __file__, ...)
            continue
        if _SENSITIVE_NAME_RE.search(name):
            out[name] = "<redacted>"
            continue
        out[name] = _safe_repr(value)
    return out


# ---------------------------------------------------------------------------
# Capture from traceback
# ---------------------------------------------------------------------------

def _frames_from_traceback(tb: TracebackType, max_frames: int = 10) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    cur: Optional[TracebackType] = tb
    while cur is not None:
        f = cur.tb_frame
        frames.append(FrameRecord(
            file=f.f_code.co_filename,
            line=cur.tb_lineno,
            function=f.f_code.co_name,
            locals=_safe_locals(f),
        ))
        cur = cur.tb_next
    if len(frames) > max_frames:
        # Keep the outermost + innermost frames (skip the middle).
        kept = frames[:1] + frames[-(max_frames - 1):]
        frames = kept
    return frames


def capture_exception(
    exc_type: type,
    exc_value: BaseException,
    tb: TracebackType,
) -> TraceRecord:
    """Build a TraceRecord from an explicit ``(type, value, tb)`` triple."""
    record = TraceRecord(
        ok=False,
        exc_type=exc_type.__name__,
        exc_message=str(exc_value),
        frames=_frames_from_traceback(tb),
    )
    return record


def capture_from_sys_exc_info() -> Optional[TraceRecord]:
    """Build a TraceRecord from ``sys.exc_info()``; returns None if no exception active."""
    et, ev, tb = sys.exc_info()
    if et is None or tb is None or ev is None:
        return None
    return capture_exception(et, ev, tb)


# ---------------------------------------------------------------------------
# Capture around a callable
# ---------------------------------------------------------------------------

def capture(fn: Callable[..., Any], *args, **kwargs) -> TraceRecord:
    """Invoke ``fn(*args, **kwargs)`` and return a TraceRecord.

    On success: ``record.ok is True`` and ``record.value`` holds the return
    value.  On exception: ``record.ok is False`` and the stack is captured.
    """
    import time
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentional broad capture
        tb = exc.__traceback__
        record = capture_exception(type(exc), exc, tb)   # type: ignore[arg-type]
        record.duration_s = time.monotonic() - t0
        return record
    return TraceRecord(
        ok=True,
        duration_s=time.monotonic() - t0,
        value=result,
    )


# ---------------------------------------------------------------------------
# pytest integration helper
# ---------------------------------------------------------------------------

def pytest_hook_from_excinfo(excinfo) -> TraceRecord:
    """Build a TraceRecord from a pytest ``ExceptionInfo`` object.

    The caller can import and use this inside a pytest ``pytest_exception_interact``
    or ``pytest_runtest_makereport`` hook.  Example::

        def pytest_exception_interact(node, call, report):
            record = pytest_hook_from_excinfo(call.excinfo)
            _stash_for_agent(record)
    """
    return capture_exception(
        excinfo.type,
        excinfo.value,
        excinfo.tb,
    )


__all__ = [
    "FrameRecord",
    "TraceRecord",
    "capture",
    "capture_exception",
    "capture_from_sys_exc_info",
    "pytest_hook_from_excinfo",
]
