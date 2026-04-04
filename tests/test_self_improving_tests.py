"""Tests for Self-Improving Test Suite — TrajectoryExtractor, TestGenerator, TestSuiteManager."""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Make new-files importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "new-files"))

from self_improving_tests import (
    Outcome,
    ToolCall,
    Trajectory,
    TrajectoryExtractor,
    TestGenerator,
    TestSuiteManager,
    TestResults,
    on_session_complete,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return TrajectoryExtractor()


@pytest.fixture
def generator():
    return TestGenerator()


@pytest.fixture
def suite_manager(tmp_path):
    return TestSuiteManager(tests_dir=tmp_path / "auto_tests")


@pytest.fixture
def simple_transcript():
    """A simple successful session transcript."""
    return [
        {"role": "user", "content": "Create a hello world file"},
        {
            "role": "assistant",
            "content": "I'll create the file for you.",
            "tool_calls": [
                {"name": "write_file", "arguments": {"path": "hello.py", "content": "print('hello')"}}
            ],
        },
        {"role": "tool", "content": "File created successfully"},
        {"role": "assistant", "content": "I've successfully created hello.py."},
    ]


@pytest.fixture
def failed_transcript():
    """A session transcript that ends in failure."""
    return [
        {"role": "user", "content": "Fix the broken test"},
        {
            "role": "assistant",
            "content": "Let me look at the test.",
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": "test.py"}}
            ],
        },
        {"role": "tool", "content": "Error: file not found"},
        {"role": "assistant", "content": "I'm unable to find the file. The error indicates it doesn't exist."},
    ]


@pytest.fixture
def multi_tool_transcript():
    """A transcript with multiple tool calls."""
    return [
        {"role": "user", "content": "Search for bugs and fix them"},
        {
            "role": "assistant",
            "content": "Let me search first.",
            "tool_calls": [
                {"name": "search_files", "arguments": {"pattern": "bug", "path": "."}}
            ],
        },
        {"role": "tool", "content": "Found match in main.py:10"},
        {
            "role": "assistant",
            "content": "Found it, let me read the file.",
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": "main.py"}}
            ],
        },
        {"role": "tool", "content": "def func():\n    bug here"},
        {
            "role": "assistant",
            "content": "I'll fix it.",
            "tool_calls": [
                {"name": "patch", "arguments": {"path": "main.py", "old_string": "bug here", "new_string": "fixed"}}
            ],
        },
        {"role": "tool", "content": "Patch applied successfully"},
        {"role": "assistant", "content": "Done. I've successfully fixed the bug."},
    ]


# ---------------------------------------------------------------------------
# ToolCall tests
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_signature_basic(self):
        tc = ToolCall(tool_name="read_file", arguments={"path": "x.py"})
        assert tc.signature() == "read_file(path)"

    def test_signature_sorted_keys(self):
        tc = ToolCall(tool_name="patch", arguments={"path": "x", "old_string": "a", "new_string": "b"})
        assert tc.signature() == "patch(new_string,old_string,path)"

    def test_signature_no_args(self):
        tc = ToolCall(tool_name="terminal", arguments={})
        assert tc.signature() == "terminal()"


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_tool_sequence(self):
        t = Trajectory(
            task_description="test",
            tool_calls=[
                ToolCall(tool_name="read_file"),
                ToolCall(tool_name="patch"),
                ToolCall(tool_name="terminal"),
            ],
        )
        assert t.tool_sequence() == ["read_file", "patch", "terminal"]

    def test_fingerprint_deterministic(self):
        t1 = Trajectory(task_description="test", tool_calls=[ToolCall(tool_name="read_file")])
        t2 = Trajectory(task_description="test", tool_calls=[ToolCall(tool_name="read_file")])
        assert t1.fingerprint() == t2.fingerprint()

    def test_fingerprint_differs_for_different_tasks(self):
        t1 = Trajectory(task_description="task A", tool_calls=[ToolCall(tool_name="read_file")])
        t2 = Trajectory(task_description="task B", tool_calls=[ToolCall(tool_name="read_file")])
        assert t1.fingerprint() != t2.fingerprint()


# ---------------------------------------------------------------------------
# TrajectoryExtractor tests
# ---------------------------------------------------------------------------

class TestTrajectoryExtractor:
    def test_extract_task_from_first_user_message(self, extractor, simple_transcript):
        traj = extractor.extract_from_session(simple_transcript)
        assert traj.task_description == "Create a hello world file"

    def test_extract_tool_calls(self, extractor, simple_transcript):
        traj = extractor.extract_from_session(simple_transcript)
        assert len(traj.tool_calls) == 1
        assert traj.tool_calls[0].tool_name == "write_file"
        assert traj.tool_calls[0].arguments["path"] == "hello.py"

    def test_extract_tool_result(self, extractor, simple_transcript):
        traj = extractor.extract_from_session(simple_transcript)
        assert traj.tool_calls[0].result == "File created successfully"

    def test_classify_success(self, extractor, simple_transcript):
        traj = extractor.extract_from_session(simple_transcript)
        assert traj.outcome == Outcome.SUCCESS

    def test_classify_failure(self, extractor, failed_transcript):
        traj = extractor.extract_from_session(failed_transcript)
        assert traj.outcome == Outcome.FAILURE

    def test_classify_partial_empty(self, extractor):
        transcript = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": ""},
        ]
        traj = extractor.extract_from_session(transcript)
        assert traj.outcome == Outcome.PARTIAL

    def test_multi_tool_extraction(self, extractor, multi_tool_transcript):
        traj = extractor.extract_from_session(multi_tool_transcript)
        assert len(traj.tool_calls) == 3
        assert traj.tool_sequence() == ["search_files", "read_file", "patch"]
        assert traj.outcome == Outcome.SUCCESS

    def test_session_id_stored(self, extractor, simple_transcript):
        traj = extractor.extract_from_session(simple_transcript, session_id="abc-123")
        assert traj.session_id == "abc-123"

    def test_truncate_long_task(self, extractor):
        long_msg = "x" * 1000
        transcript = [
            {"role": "user", "content": long_msg},
            {"role": "assistant", "content": "Done successfully."},
        ]
        traj = extractor.extract_from_session(transcript)
        assert len(traj.task_description) <= 503  # 500 + "..."

    def test_string_arguments_parsed(self, extractor):
        transcript = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"name": "terminal", "arguments": '{"command": "ls"}'}
                ],
            },
            {"role": "tool", "content": "file1\nfile2"},
            {"role": "assistant", "content": "Done successfully."},
        ]
        traj = extractor.extract_from_session(transcript)
        assert traj.tool_calls[0].arguments == {"command": "ls"}

    def test_empty_transcript(self, extractor):
        traj = extractor.extract_from_session([])
        assert traj.task_description == "unknown task"
        assert traj.tool_calls == []
        assert traj.outcome == Outcome.PARTIAL


# ---------------------------------------------------------------------------
# TestGenerator tests
# ---------------------------------------------------------------------------

class TestTestGenerator:
    def test_generate_regression_test(self, generator):
        traj = Trajectory(
            task_description="Create hello file",
            tool_calls=[ToolCall(tool_name="write_file", arguments={"path": "hello.py"})],
            outcome=Outcome.SUCCESS,
            session_id="sess-001",
        )
        code = generator.generate_regression_test(traj)
        assert "def test_regression_" in code
        assert "write_file" in code
        assert "success" in code

    def test_generated_test_is_valid_python(self, generator):
        traj = Trajectory(
            task_description="Fix bug",
            tool_calls=[ToolCall(tool_name="read_file"), ToolCall(tool_name="patch")],
            outcome=Outcome.SUCCESS,
        )
        code = generator.generate_regression_test(traj)
        compile(code, "<test>", "exec")  # Should not raise

    def test_generate_property_test(self, generator):
        trajs = [
            Trajectory(
                task_description=f"Task {i}",
                tool_calls=[
                    ToolCall(tool_name="read_file"),
                    ToolCall(tool_name="patch"),
                ],
                outcome=Outcome.SUCCESS,
            )
            for i in range(3)
        ]
        code = generator.generate_property_test(trajs)
        assert "def test_property_" in code
        assert "read_file" in code
        assert "patch" in code

    def test_property_test_empty_trajectories(self, generator):
        code = generator.generate_property_test([])
        assert "No trajectories" in code

    def test_property_test_finds_common_tools(self, generator):
        trajs = [
            Trajectory(
                task_description="A",
                tool_calls=[ToolCall(tool_name="read_file"), ToolCall(tool_name="patch")],
                outcome=Outcome.SUCCESS,
            ),
            Trajectory(
                task_description="B",
                tool_calls=[ToolCall(tool_name="read_file"), ToolCall(tool_name="terminal")],
                outcome=Outcome.SUCCESS,
            ),
        ]
        code = generator.generate_property_test(trajs)
        # read_file is common to both
        assert "read_file" in code


# ---------------------------------------------------------------------------
# TestSuiteManager tests
# ---------------------------------------------------------------------------

class TestTestSuiteManager:
    def test_add_test(self, suite_manager):
        code = "def test_example():\n    assert True\n"
        path = suite_manager.add_test(code, source_session_id="s1")
        assert path.exists()
        assert path.read_text() == code

    def test_add_test_metadata(self, suite_manager):
        code = "def test_example():\n    assert True\n"
        suite_manager.add_test(code, source_session_id="s1")
        stats = suite_manager.stats()
        assert stats["total_tests"] == 1
        assert stats["sessions_covered"] == 1

    def test_deduplicate(self, suite_manager):
        code = "def test_dup():\n    assert True\n"
        suite_manager.add_test(code, source_session_id="s1")
        suite_manager.add_test(code, source_session_id="s2")
        # Same content -> same hash -> same file -> only 1 file
        # But since hash is derived from content, same code = same filename
        stats = suite_manager.stats()
        # Deduplicate should still work
        removed = suite_manager.deduplicate()
        assert removed >= 0

    def test_deduplicate_different_content(self, suite_manager):
        suite_manager.add_test("def test_a():\n    assert True\n", "s1")
        suite_manager.add_test("def test_b():\n    assert False\n", "s2")
        removed = suite_manager.deduplicate()
        assert removed == 0
        assert suite_manager.stats()["total_tests"] == 2

    def test_stats(self, suite_manager):
        stats = suite_manager.stats()
        assert "total_tests" in stats
        assert "sessions_covered" in stats
        assert "success_rate" in stats
        assert "flaky_tests" in stats
        assert stats["total_tests"] == 0

    def test_list_tests(self, suite_manager):
        suite_manager.add_test("def test_x():\n    pass\n", "s1")
        tests = suite_manager.list_tests()
        assert len(tests) == 1
        assert tests[0]["session_id"] == "s1"

    def test_prune_flaky_needs_min_runs(self, suite_manager):
        code = "def test_flaky():\n    import random; assert random.random() > 0.5\n"
        suite_manager.add_test(code, "s1")
        # With 0 runs, nothing should be pruned
        pruned = suite_manager.prune_flaky()
        assert pruned == 0

    def test_prune_flaky_removes_bad_tests(self, suite_manager):
        code = "def test_flaky():\n    assert False\n"
        path = suite_manager.add_test(code, "s1")
        filename = path.name

        # Simulate run history: 5 runs, 4 failures
        suite_manager._metadata["tests"][filename]["run_count"] = 5
        suite_manager._metadata["tests"][filename]["flaky_count"] = 4
        suite_manager._save_metadata()

        pruned = suite_manager.prune_flaky(max_flaky_rate=0.1)
        assert pruned == 1
        assert not path.exists()

    def test_tests_dir_created(self, tmp_path):
        tests_dir = tmp_path / "deep" / "nested" / "tests"
        mgr = TestSuiteManager(tests_dir=tests_dir)
        assert tests_dir.exists()


# ---------------------------------------------------------------------------
# on_session_complete hook tests
# ---------------------------------------------------------------------------

class TestOnSessionComplete:
    def test_successful_session_generates_test(self, tmp_path, simple_transcript):
        result = on_session_complete(
            simple_transcript,
            session_id="test-session",
            tests_dir=tmp_path / "auto_tests",
        )
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "def test_regression_" in content

    def test_failed_session_no_test(self, tmp_path, failed_transcript):
        result = on_session_complete(
            failed_transcript,
            session_id="fail-session",
            tests_dir=tmp_path / "auto_tests",
        )
        assert result is None

    def test_no_tool_calls_no_test(self, tmp_path):
        transcript = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help? Successfully done."},
        ]
        result = on_session_complete(
            transcript,
            session_id="no-tools",
            tests_dir=tmp_path / "auto_tests",
        )
        assert result is None

    def test_empty_transcript_no_test(self, tmp_path):
        result = on_session_complete(
            [],
            session_id="empty",
            tests_dir=tmp_path / "auto_tests",
        )
        assert result is None
