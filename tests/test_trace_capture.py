"""Tests for ``trace_capture``."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from trace_capture import (  # noqa: E402
    capture,
    capture_exception,
    capture_from_sys_exc_info,
)

# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------

class TestCapture:
    def test_happy_path(self):
        rec = capture(lambda x: x + 1, 41)
        assert rec.ok
        assert rec.value == 42
        assert rec.exc_type is None
        assert rec.frames == []

    def test_captures_exception(self):
        def boom():
            x = 5
            secret = "hunter2"   # noqa: F841 - should be redacted
            raise ValueError("oops")

        rec = capture(boom)
        assert not rec.ok
        assert rec.exc_type == "ValueError"
        assert "oops" in (rec.exc_message or "")
        assert rec.frames, "expected at least one frame"

    def test_local_redaction(self):
        def boom():
            password = "hunter2"   # noqa: F841
            api_key = "sk_live_xyz"  # noqa: F841
            raise RuntimeError("x")

        rec = capture(boom)
        bottom = rec.bottom_frame
        assert bottom is not None
        assert bottom.locals.get("password") == "<redacted>"
        assert bottom.locals.get("api_key") == "<redacted>"

    def test_local_values_included(self):
        def boom():
            answer = 42
            phrase = "hello"  # noqa: F841
            raise ZeroDivisionError("x")

        rec = capture(boom)
        bottom = rec.bottom_frame
        assert bottom is not None
        assert bottom.locals.get("answer") == "42"
        assert bottom.locals.get("phrase") == "'hello'"

    def test_repr_of_broken_object(self):
        class Broken:
            def __repr__(self):
                raise RuntimeError("repr blew up")

        def boom():
            broken = Broken()   # noqa: F841
            raise ValueError("x")

        rec = capture(boom)
        bottom = rec.bottom_frame
        assert bottom is not None
        # Value repr is safe and marked as a repr-error.
        assert bottom.locals.get("broken", "").startswith("<repr-error")


# ---------------------------------------------------------------------------
# capture_exception() from an existing try/except
# ---------------------------------------------------------------------------

class TestCaptureExceptionDirect:
    def test_captures_current_exception(self):
        try:
            int("hello")
        except Exception as exc:
            rec = capture_exception(type(exc), exc, exc.__traceback__)
        assert rec.exc_type == "ValueError"
        assert rec.frames

    def test_capture_from_sys_exc_info_returns_none_when_no_exc(self):
        assert capture_from_sys_exc_info() is None


# ---------------------------------------------------------------------------
# Long-repr truncation and big-locals cap
# ---------------------------------------------------------------------------

class TestBoundedRepr:
    def test_long_repr_truncated(self):
        def boom():
            big = "x" * 10_000   # noqa: F841
            raise RuntimeError("x")

        rec = capture(boom)
        bottom = rec.bottom_frame
        val = bottom.locals.get("big", "")
        assert val.endswith("...")
        assert len(val) < 200

    def test_max_locals_enforced(self):
        def boom():
            ns = {f"var_{i}": i for i in range(100)}
            locals().update(ns)   # inject into frame.f_locals
            raise RuntimeError("x")
        rec = capture(boom)
        bottom = rec.bottom_frame
        assert bottom is not None
        # The capture never keeps more than 32 locals + truncation marker.
        assert len(bottom.locals) <= 33


# ---------------------------------------------------------------------------
# format_agent_context
# ---------------------------------------------------------------------------

class TestFormat:
    def test_format_contains_exception_type_and_frame(self):
        def boom():
            x = 10  # noqa: F841
            raise ValueError("something went wrong")
        rec = capture(boom)
        out = rec.format_agent_context()
        assert "ValueError" in out
        assert "something went wrong" in out
        assert "in boom()" in out

    def test_format_respects_max_frames(self):
        def level_1():
            raise RuntimeError("deep")
        def level_2():
            level_1()
        def level_3():
            level_2()
        rec = capture(level_3)
        out = rec.format_agent_context(max_frames=1)
        # Only one frame rendered despite deeper stack.
        assert out.count("in level_") == 1


# ---------------------------------------------------------------------------
# Frame count cap
# ---------------------------------------------------------------------------

class TestFrameCap:
    def test_frames_capped(self):
        def deep(n):
            if n <= 0:
                raise RuntimeError("bottom")
            deep(n - 1)
        rec = capture(deep, 20)
        assert not rec.ok
        # Module defaults to 10 frames retained.
        assert len(rec.frames) <= 10
