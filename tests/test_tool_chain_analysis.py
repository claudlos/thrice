"""Tests for tool_chain_analysis.py — Causal Inference on Tool Chains (#13)."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from tool_chain_analysis import (
    ToolChainLogger, CausalAnalyzer, LoopDetector, Pattern, ChainRecord
)


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def logger(tmp_db):
    return ToolChainLogger(db_path=tmp_db)


@pytest.fixture
def populated_logger(logger):
    """Logger with 60+ synthetic chains for DAG building."""
    # Good patterns: search -> read -> edit -> test (high success)
    for i in range(25):
        logger.log_chain(f"s{i}", ["search", "read", "edit", "test"], "success", 10.0 + i)
    # Bad patterns: edit -> test -> edit -> test (loop, low success)
    for i in range(15):
        logger.log_chain(f"b{i}", ["edit", "test", "edit", "test"], "failure", 30.0 + i)
    # Mixed: search -> edit -> test
    for i in range(10):
        outcome = "success" if i % 2 == 0 else "failure"
        logger.log_chain(f"m{i}", ["search", "edit", "test"], outcome, 15.0)
    # Bad: read -> read -> read (stuck)
    for i in range(10):
        logger.log_chain(f"r{i}", ["read", "read", "read"], "failure", 20.0)
    # Extra for min_chains
    for i in range(5):
        logger.log_chain(f"e{i}", ["search", "read", "test"], "success", 8.0)
    return logger


class TestToolChainLogger:
    def test_log_and_retrieve(self, logger):
        row_id = logger.log_chain("sess1", ["search", "read"], "success", 5.0)
        assert row_id is not None
        chains = logger.get_chains()
        assert len(chains) == 1
        assert chains[0].session_id == "sess1"
        assert chains[0].tool_sequence == ["search", "read"]
        assert chains[0].outcome == "success"

    def test_invalid_outcome(self, logger):
        with pytest.raises(ValueError, match="outcome must be"):
            logger.log_chain("s", ["tool"], "maybe", 1.0)

    def test_empty_sequence(self, logger):
        with pytest.raises(ValueError, match="empty"):
            logger.log_chain("s", [], "success", 1.0)

    def test_count(self, logger):
        assert logger.count_chains() == 0
        logger.log_chain("s1", ["a"], "success", 1.0)
        logger.log_chain("s2", ["b"], "failure", 2.0)
        assert logger.count_chains() == 2

    def test_filter_by_outcome(self, logger):
        logger.log_chain("s1", ["a"], "success", 1.0)
        logger.log_chain("s2", ["b"], "failure", 2.0)
        successes = logger.get_chains(outcome="success")
        assert len(successes) == 1
        assert successes[0].outcome == "success"


class TestCausalAnalyzer:
    def test_build_dag_insufficient_data(self, logger):
        logger.log_chain("s1", ["a", "b"], "success", 1.0)
        analyzer = CausalAnalyzer(logger)
        dag = analyzer.build_dag(min_chains=50)
        assert dag == {}

    def test_build_dag_with_data(self, populated_logger):
        analyzer = CausalAnalyzer(populated_logger)
        dag = analyzer.build_dag(min_chains=50)
        assert dag  # Not empty
        assert "search" in dag
        assert "read" in dag["search"]

    def test_compute_success_rate(self, populated_logger):
        analyzer = CausalAnalyzer(populated_logger)
        analyzer.build_dag(min_chains=50)
        # Good pattern should have high success rate
        rate = analyzer.compute_success_rate(["search", "read", "edit", "test"])
        assert rate > 0.7
        # Bad pattern should have low success rate
        rate = analyzer.compute_success_rate(["edit", "test", "edit", "test"])
        assert rate < 0.4

    def test_find_bad_patterns(self, populated_logger):
        analyzer = CausalAnalyzer(populated_logger)
        analyzer.build_dag(min_chains=50)
        bad = analyzer.find_bad_patterns()
        assert len(bad) > 0
        for p in bad:
            assert p.success_rate < 0.4
            assert isinstance(p, Pattern)

    def test_find_optimal_orderings(self, populated_logger):
        analyzer = CausalAnalyzer(populated_logger)
        analyzer.build_dag(min_chains=50)
        good = analyzer.find_optimal_orderings()
        assert len(good) > 0
        for p in good:
            assert p.success_rate > 0.7

    def test_detect_loop(self, populated_logger):
        analyzer = CausalAnalyzer(populated_logger)
        analyzer.build_dag(min_chains=50)
        # A chain repeating the same tool should be detected
        assert analyzer.detect_loop(["read", "read", "read"]) is True
        # A normal chain should not
        assert analyzer.detect_loop(["search", "read"]) is False


class TestLoopDetector:
    def test_consecutive_identical(self):
        detector = LoopDetector()
        assert detector.is_stuck(["read", "read", "read"], []) is True
        assert detector.is_stuck(["read", "write", "read"], []) is False

    def test_custom_threshold(self):
        detector = LoopDetector(repeat_threshold=2)
        assert detector.is_stuck(["read", "read"], []) is True

    def test_error_pattern(self):
        detector = LoopDetector()
        tools = ["edit", "test", "edit", "test"]
        errors = ["", "fail", "", "fail"]
        assert detector.is_stuck(tools, errors) is True

    def test_high_frequency_tool(self):
        detector = LoopDetector()
        tools = ["search", "read", "search", "read", "search", "read", "search", "read", "search", "read"]
        assert detector.is_stuck(tools, []) is True  # search appears 5 times in 10

    def test_empty_input(self):
        detector = LoopDetector()
        assert detector.is_stuck([], []) is False

    def test_suggest_escape_repeated(self):
        detector = LoopDetector()
        chain = ["search", "search", "search", "search"]
        suggestion = detector.suggest_escape(chain)
        assert "search" in suggestion.lower()
        assert len(suggestion) > 10

    def test_suggest_escape_alternating(self):
        detector = LoopDetector()
        chain = ["read", "write", "read", "write", "read", "write"]
        suggestion = detector.suggest_escape(chain)
        assert len(suggestion) > 10

    def test_suggest_escape_empty(self):
        detector = LoopDetector()
        suggestion = detector.suggest_escape([])
        assert "different approach" in suggestion.lower()
