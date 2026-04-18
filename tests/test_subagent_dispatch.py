"""Tests for ``subagent_dispatch``."""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from subagent_dispatch import (                        # noqa: E402
    SubTask,
    SubagentDispatcher,
    as_runner,
    inspect_pattern,
    search_pattern,
    summarize_pattern,
)


# ---------------------------------------------------------------------------
# Canonical patterns generate well-formed SubTasks
# ---------------------------------------------------------------------------

class TestPatterns:
    def test_search_pattern_shape(self):
        t = search_pattern("TODO", paths=["src/", "tests/"])
        assert t.kind == "search"
        assert "TODO" in t.prompt
        assert t.context == {"paths": ["src/", "tests/"]}

    def test_summarize_pattern_with_focus(self):
        t = summarize_pattern("long artifact here", focus="error handling")
        assert "error handling" in t.prompt
        assert "long artifact here" in t.prompt

    def test_inspect_pattern_asks_json(self):
        t = inspect_pattern("Does X import Y?", "file contents")
        assert t.kind == "inspect"
        assert "JSON" in t.prompt
        assert "Does X import Y?" in t.prompt


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestDispatchHappyPath:

    def test_run_returns_structured_result(self):
        runner = as_runner(lambda t: f"echo: {t.prompt[:30]}")
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="test", prompt="hello subagent"))
        assert r.ok
        assert "hello" in r.summary
        assert r.cost_tokens > 0
        assert r.duration_s >= 0
        assert r.error is None

    def test_stats_tracks_successes(self):
        runner = as_runner(lambda t: "ok")
        with SubagentDispatcher(runner) as d:
            for _ in range(5):
                d.run(SubTask(kind="test", prompt="x"))
        assert d.stats.dispatched == 5
        assert d.stats.succeeded == 5
        assert d.stats.failed == 0
        assert d.stats.total_tokens > 0


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetCaps:
    def test_input_token_cap(self):
        runner = as_runner(lambda t: "should not run")
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(
                kind="test",
                prompt="x" * 40_000,         # ~10k tokens
                max_input_tokens=1000,
            ))
        assert not r.ok
        assert r.error and "input tokens" in r.error

    def test_output_truncation(self):
        big = "z" * 10_000
        runner = as_runner(lambda t: big)
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(
                kind="test",
                prompt="hi",
                truncate_output_chars=500,
            ))
        assert r.ok
        assert r.truncated
        assert len(r.raw) == 500

    def test_timeout(self):
        def slow_runner(task):
            time.sleep(1.5)
            return "done", 1, 1
        with SubagentDispatcher(slow_runner) as d:
            r = d.run(SubTask(kind="test", prompt="hi", timeout_s=0.1))
        assert not r.ok
        assert r.error and "timed out" in r.error
        assert d.stats.timed_out == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_runner_exception_captured(self):
        def boom_runner(task):
            raise RuntimeError("boom")
        with SubagentDispatcher(boom_runner) as d:
            r = d.run(SubTask(kind="test", prompt="hi"))
        assert not r.ok
        assert r.error and "RuntimeError" in r.error
        assert "boom" in r.error


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------

class TestFanOut:
    def test_run_many_preserves_order(self):
        runner = as_runner(lambda t: t.prompt.upper())
        with SubagentDispatcher(runner) as d:
            tasks = [SubTask(kind="test", prompt=f"p{i}") for i in range(5)]
            results = d.run_many(tasks)
        assert len(results) == 5
        assert [r.summary for r in results] == ["P0", "P1", "P2", "P3", "P4"]


# ---------------------------------------------------------------------------
# Summary / format-for-caller
# ---------------------------------------------------------------------------

class TestSummary:
    def test_search_keeps_line_shape(self):
        out = "src/a.py:10: TODO fix\nsrc/b.py:42: TODO me\n"
        runner = as_runner(lambda t: out)
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="search", prompt="find TODO"))
        assert "src/a.py" in r.summary
        assert "src/b.py" in r.summary

    def test_inspect_parses_json_when_available(self):
        runner = as_runner(
            lambda t: '{"answer":"yes","reason":"imports X","evidence":"line 3"}',
        )
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="inspect", prompt="?"))
        data = json.loads(r.summary)
        assert data["answer"] == "yes"

    def test_inspect_falls_back_when_not_json(self):
        runner = as_runner(lambda t: "yes, definitely.")
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="inspect", prompt="?"))
        # Non-JSON inspect result is still surfaced (no parse error).
        assert r.summary

    def test_format_for_caller_compact(self):
        runner = as_runner(lambda t: "x" * 2000)
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="test", prompt="hi"))
        block = r.format_for_caller(max_chars=100)
        assert block.startswith("[subagent:ok")
        assert len(block) < 200


# ---------------------------------------------------------------------------
# Integration with cost_estimator
# ---------------------------------------------------------------------------

class TestCostIntegration:
    def test_cost_is_recordable(self, tmp_path):
        """Sub-agent cost can feed into cost_estimator for future budgeting."""
        from cost_estimator import CostEstimator, TaskSpec

        est = CostEstimator(history_path=str(tmp_path / "h.jsonl"))

        runner = as_runner(lambda t: "ok")
        with SubagentDispatcher(runner) as d:
            r = d.run(SubTask(kind="search", prompt="find TODO"))

        est.record(
            TaskSpec(kind="subagent:search"),
            tokens=r.cost_tokens,
            iterations=1,
            wall_s=r.duration_s,
        )
        recs = est._load()
        assert len(recs) == 1
        assert recs[0].tokens == r.cost_tokens
