"""
Tests for tool_call_batcher.py — Intelligent Tool Call Batching.

30+ tests covering ToolCallRecord, DependencyGraph, DependencyAnalyzer,
PrefetchPredictor, PrefetchCache, and BatchingAdvisor.
"""

import sys
import os
import time
import pytest

# Ensure the new-files directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from tool_call_batcher import (
    ToolCallRecord,
    ToolCategory,
    DependencyGraph,
    DependencyAnalyzer,
    PrefetchPredictor,
    PrefetchCache,
    BatchingAdvisor,
    _categorize,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _rec(name, args=None, result=None, call_id=None, ts=None):
    """Shorthand for creating a ToolCallRecord."""
    return ToolCallRecord(
        tool_name=name,
        args=args or {},
        result=result,
        call_id=call_id or "",
        timestamp=ts or time.time(),
    )


# -----------------------------------------------------------------------
# ToolCallRecord
# -----------------------------------------------------------------------

class TestToolCallRecord:

    def test_auto_id(self):
        rec = _rec("read_file")
        assert rec.call_id  # non-empty
        assert len(rec.call_id) == 12

    def test_explicit_id(self):
        rec = _rec("read_file", call_id="abc123")
        assert rec.call_id == "abc123"

    def test_auto_timestamp(self):
        before = time.time()
        rec = ToolCallRecord(tool_name="x", args={})
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_get_target_path(self):
        rec = _rec("read_file", args={"path": "/foo/bar.py"})
        assert rec.get_target_path() == "/foo/bar.py"

    def test_get_target_path_file_key(self):
        rec = _rec("read_file", args={"file": "/a/b.py"})
        assert rec.get_target_path() == "/a/b.py"

    def test_get_target_path_none(self):
        rec = _rec("terminal", args={"command": "ls"})
        assert rec.get_target_path() is None

    def test_depends_on_default(self):
        rec = _rec("read_file")
        assert rec.depends_on == []


# -----------------------------------------------------------------------
# _categorize
# -----------------------------------------------------------------------

class TestCategorize:

    def test_read(self):
        assert _categorize("read_file") == ToolCategory.READ
        assert _categorize("mcp_read_file") == ToolCategory.READ

    def test_write(self):
        assert _categorize("write_file") == ToolCategory.WRITE
        assert _categorize("mcp_patch") == ToolCategory.WRITE

    def test_search(self):
        assert _categorize("search_files") == ToolCategory.SEARCH

    def test_execute(self):
        assert _categorize("terminal") == ToolCategory.EXECUTE

    def test_unknown(self):
        assert _categorize("foobar") == ToolCategory.OTHER


# -----------------------------------------------------------------------
# DependencyGraph
# -----------------------------------------------------------------------

class TestDependencyGraph:

    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.get_execution_order() == []
        assert len(g) == 0

    def test_single_node(self):
        g = DependencyGraph()
        r = _rec("read_file", call_id="a")
        g.add_node(r)
        order = g.get_execution_order()
        assert len(order) == 1
        assert order[0] == [r]

    def test_two_independent_nodes(self):
        g = DependencyGraph()
        a = _rec("read_file", call_id="a", ts=1.0)
        b = _rec("read_file", call_id="b", ts=2.0)
        g.add_node(a)
        g.add_node(b)
        order = g.get_execution_order()
        # Both in same group
        assert len(order) == 1
        assert set(r.call_id for r in order[0]) == {"a", "b"}

    def test_chain_dependency(self):
        g = DependencyGraph()
        a = _rec("read_file", call_id="a", ts=1.0)
        b = _rec("write_file", call_id="b", ts=2.0)
        g.add_node(a)
        g.add_node(b)
        g.add_edge("a", "b")
        order = g.get_execution_order()
        assert len(order) == 2
        assert order[0][0].call_id == "a"
        assert order[1][0].call_id == "b"

    def test_diamond_dependency(self):
        g = DependencyGraph()
        a = _rec("x", call_id="a", ts=1.0)
        b = _rec("x", call_id="b", ts=2.0)
        c = _rec("x", call_id="c", ts=3.0)
        d = _rec("x", call_id="d", ts=4.0)
        for n in [a, b, c, d]:
            g.add_node(n)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "d")
        g.add_edge("c", "d")
        order = g.get_execution_order()
        assert len(order) == 3
        assert order[0][0].call_id == "a"
        assert set(r.call_id for r in order[1]) == {"b", "c"}
        assert order[2][0].call_id == "d"

    def test_get_dependencies(self):
        g = DependencyGraph()
        a = _rec("x", call_id="a")
        b = _rec("x", call_id="b")
        g.add_node(a)
        g.add_node(b)
        g.add_edge("a", "b")
        assert g.get_dependencies("b") == {"a"}
        assert g.get_dependencies("a") == set()

    def test_get_dependents(self):
        g = DependencyGraph()
        a = _rec("x", call_id="a")
        b = _rec("x", call_id="b")
        g.add_node(a)
        g.add_node(b)
        g.add_edge("a", "b")
        assert g.get_dependents("a") == {"b"}


# -----------------------------------------------------------------------
# DependencyAnalyzer
# -----------------------------------------------------------------------

class TestDependencyAnalyzer:

    def setup_method(self):
        self.analyzer = DependencyAnalyzer()

    def test_reads_are_independent(self):
        a = _rec("read_file", args={"path": "/a.py"})
        b = _rec("read_file", args={"path": "/b.py"})
        assert self.analyzer.are_independent(a, b) is True

    def test_searches_are_independent(self):
        a = _rec("search_files", args={"pattern": "foo"})
        b = _rec("search_files", args={"pattern": "bar"})
        assert self.analyzer.are_independent(a, b) is True

    def test_read_and_search_independent(self):
        a = _rec("read_file", args={"path": "/x.py"})
        b = _rec("search_files", args={"pattern": "x"})
        assert self.analyzer.are_independent(a, b) is True

    def test_write_depends_on_read_same_file(self):
        a = _rec("read_file", args={"path": "/x.py"})
        b = _rec("write_file", args={"path": "/x.py"})
        assert self.analyzer.are_independent(a, b) is False

    def test_write_independent_of_read_different_file(self):
        a = _rec("read_file", args={"path": "/a.py"})
        b = _rec("write_file", args={"path": "/b.py"})
        assert self.analyzer.are_independent(a, b) is True

    def test_two_writes_same_file_dependent(self):
        a = _rec("write_file", args={"path": "/x.py"})
        b = _rec("write_file", args={"path": "/x.py"})
        assert self.analyzer.are_independent(a, b) is False

    def test_terminals_dependent(self):
        a = _rec("terminal", args={"command": "ls"})
        b = _rec("terminal", args={"command": "pwd"})
        assert self.analyzer.are_independent(a, b) is False

    def test_write_and_terminal_dependent(self):
        a = _rec("write_file", args={"path": "/x.py"})
        b = _rec("terminal", args={"command": "python /x.py"})
        assert self.analyzer.are_independent(a, b) is False

    def test_analyze_produces_graph(self):
        calls = [
            _rec("read_file", args={"path": "/a.py"}, call_id="r1", ts=1.0),
            _rec("read_file", args={"path": "/b.py"}, call_id="r2", ts=2.0),
            _rec("write_file", args={"path": "/a.py"}, call_id="w1", ts=3.0),
        ]
        graph = self.analyzer.analyze_dependencies(calls)
        assert len(graph) == 3
        # w1 depends on r1 (same file), but not on r2
        assert "r1" in graph.get_dependencies("w1")

    def test_find_parallelizable_groups(self):
        calls = [
            _rec("read_file", args={"path": "/a.py"}, call_id="r1", ts=1.0),
            _rec("read_file", args={"path": "/b.py"}, call_id="r2", ts=2.0),
            _rec("search_files", args={"pattern": "x"}, call_id="s1", ts=3.0),
        ]
        groups = self.analyzer.find_parallelizable_groups(calls)
        # All three should be in one group (all independent)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_mixed_parallelism(self):
        calls = [
            _rec("read_file", args={"path": "/a.py"}, call_id="r1", ts=1.0),
            _rec("read_file", args={"path": "/b.py"}, call_id="r2", ts=2.0),
            _rec("write_file", args={"path": "/a.py"}, call_id="w1", ts=3.0),
            _rec("write_file", args={"path": "/b.py"}, call_id="w2", ts=4.0),
        ]
        groups = self.analyzer.find_parallelizable_groups(calls)
        # r1 and r2 can be parallel; w1 and w2 depend on r1/r2 respectively
        # but w1 and w2 target different files so they could be parallel
        assert len(groups) >= 2
        first_ids = {r.call_id for r in groups[0]}
        assert "r1" in first_ids and "r2" in first_ids


# -----------------------------------------------------------------------
# PrefetchPredictor
# -----------------------------------------------------------------------

class TestPrefetchPredictor:

    def setup_method(self):
        self.predictor = PrefetchPredictor()

    def test_empty_history(self):
        assert self.predictor.predict_next_tools([]) == []

    def test_search_predicts_read(self):
        history = [_rec("search_files", args={"pattern": "foo"})]
        preds = self.predictor.predict_next_tools(history)
        tool_names = [p["tool_name"] for p in preds]
        assert "read_file" in tool_names

    def test_write_predicts_terminal(self):
        history = [_rec("write_file", args={"path": "/x.py"})]
        preds = self.predictor.predict_next_tools(history)
        tool_names = [p["tool_name"] for p in preds]
        assert "terminal" in tool_names

    def test_confidence_range(self):
        history = [_rec("search_files")]
        preds = self.predictor.predict_next_tools(history)
        for p in preds:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_import_following(self):
        code = "import os\nfrom pathlib import Path\nimport json"
        history = [_rec("read_file", args={"path": "/x.py"}, result=code)]
        preds = self.predictor.predict_next_tools(history)
        # Should predict read_file and include import-follow reason
        reasons = [p.get("reason", "") for p in preds]
        assert any("import follow" in r or "read" in r for r in reasons)

    def test_mcp_variants(self):
        history = [_rec("mcp_search_files", args={"pattern": "x"})]
        preds = self.predictor.predict_next_tools(history)
        tool_names = [p["tool_name"] for p in preds]
        assert "mcp_read_file" in tool_names

    def test_patch_predicts_terminal(self):
        history = [_rec("patch", args={"path": "/x.py"})]
        preds = self.predictor.predict_next_tools(history)
        tool_names = [p["tool_name"] for p in preds]
        assert "terminal" in tool_names

    def test_learn_transitions(self):
        history = [
            _rec("read_file", ts=1.0),
            _rec("search_files", ts=2.0),
            _rec("read_file", ts=3.0),
            _rec("search_files", ts=4.0),
            _rec("read_file", ts=5.0),
            _rec("search_files", ts=6.0),
        ]
        self.predictor.learn(history)
        preds = self.predictor.predict_next_tools([_rec("read_file")])
        tool_names = [p["tool_name"] for p in preds]
        assert "search_files" in tool_names

    def test_predictions_sorted_by_confidence(self):
        history = [_rec("search_files", result="/a.py\n/b.py")]
        preds = self.predictor.predict_next_tools(history)
        if len(preds) >= 2:
            for i in range(len(preds) - 1):
                assert preds[i]["confidence"] >= preds[i + 1]["confidence"]

    def test_custom_patterns(self):
        pred = PrefetchPredictor(custom_patterns=[
            ("my_tool", "my_other_tool", 0.90),
        ])
        history = [_rec("my_tool")]
        preds = pred.predict_next_tools(history)
        tool_names = [p["tool_name"] for p in preds]
        assert "my_other_tool" in tool_names

    def test_write_py_includes_compile_command(self):
        history = [_rec("write_file", args={"path": "/foo/bar.py"})]
        preds = self.predictor.predict_next_tools(history)
        terminal_preds = [p for p in preds if p["tool_name"] == "terminal"]
        assert len(terminal_preds) >= 1
        assert "py_compile" in terminal_preds[0]["args"].get("command", "")


# -----------------------------------------------------------------------
# PrefetchCache
# -----------------------------------------------------------------------

class TestPrefetchCache:

    def test_put_get(self):
        cache = PrefetchCache()
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_miss(self):
        cache = PrefetchCache()
        assert cache.get("nope") is None
        assert cache.misses == 1

    def test_hit_tracking(self):
        cache = PrefetchCache()
        cache.put("k", "v")
        cache.get("k")
        cache.get("k")
        assert cache.hits == 2
        assert cache.misses == 0

    def test_ttl_expiry(self):
        cache = PrefetchCache(default_ttl=0.01)
        cache.put("k", "v")
        time.sleep(0.02)
        assert cache.get("k") is None
        assert cache.misses == 1

    def test_custom_ttl(self):
        cache = PrefetchCache(default_ttl=100)
        cache.put("k", "v", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_invalidate(self):
        cache = PrefetchCache()
        cache.put("k", "v")
        assert cache.invalidate("k") is True
        assert cache.get("k") is None
        assert cache.invalidate("k") is False

    def test_clear(self):
        cache = PrefetchCache()
        cache.put("a", "1")
        cache.put("b", "2")
        cache.clear()
        assert cache.size == 0

    def test_prune_expired(self):
        cache = PrefetchCache(default_ttl=0.01)
        cache.put("a", "1")
        cache.put("b", "2")
        time.sleep(0.02)
        cache.put("c", "3")  # This one is fresh
        removed = cache.prune_expired()
        assert removed == 2
        assert cache.size == 1

    def test_size(self):
        cache = PrefetchCache()
        assert cache.size == 0
        cache.put("a", "1")
        assert cache.size == 1

    def test_hit_rate(self):
        cache = PrefetchCache()
        cache.put("a", "1")
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == 0.5

    def test_hit_rate_empty(self):
        cache = PrefetchCache()
        assert cache.hit_rate == 0.0

    def test_stats(self):
        cache = PrefetchCache()
        cache.put("a", "1")
        cache.get("a")
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 0
        assert s["size"] == 1

    def test_make_key(self):
        k1 = PrefetchCache.make_key("read_file", {"path": "/a.py"})
        k2 = PrefetchCache.make_key("read_file", {"path": "/a.py"})
        k3 = PrefetchCache.make_key("read_file", {"path": "/b.py"})
        assert k1 == k2
        assert k1 != k3

    def test_make_key_sorted(self):
        k1 = PrefetchCache.make_key("t", {"a": 1, "b": 2})
        k2 = PrefetchCache.make_key("t", {"b": 2, "a": 1})
        assert k1 == k2


# -----------------------------------------------------------------------
# BatchingAdvisor
# -----------------------------------------------------------------------

class TestBatchingAdvisor:

    def setup_method(self):
        self.advisor = BatchingAdvisor()

    def test_should_batch_single(self):
        assert self.advisor.should_batch([{"tool_name": "read_file"}]) is False

    def test_should_batch_empty(self):
        assert self.advisor.should_batch([]) is False

    def test_should_batch_two_reads(self):
        calls = [
            {"tool_name": "read_file", "args": {"path": "/a.py"}},
            {"tool_name": "read_file", "args": {"path": "/b.py"}},
        ]
        assert self.advisor.should_batch(calls) is True

    def test_should_not_batch_dependent(self):
        calls = [
            {"tool_name": "terminal", "args": {"command": "ls"}},
            {"tool_name": "terminal", "args": {"command": "pwd"}},
        ]
        # Two terminals are dependent, so single-group batching isn't useful
        assert self.advisor.should_batch(calls) is False

    def test_suggest_batch_empty(self):
        assert self.advisor.suggest_batch([]) == []

    def test_suggest_batch_parallel_reads(self):
        calls = [
            {"tool_name": "read_file", "args": {"path": "/a.py"}},
            {"tool_name": "read_file", "args": {"path": "/b.py"}},
            {"tool_name": "search_files", "args": {"pattern": "x"}},
        ]
        batches = self.advisor.suggest_batch(calls)
        # All independent, should be one batch
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_suggest_batch_with_dependency(self):
        calls = [
            {"tool_name": "read_file", "args": {"path": "/a.py"}},
            {"tool_name": "write_file", "args": {"path": "/a.py"}},
        ]
        batches = self.advisor.suggest_batch(calls)
        assert len(batches) == 2

    def test_estimate_speedup_single_batch(self):
        plan = [[
            {"tool_name": "read_file"},
            {"tool_name": "read_file"},
            {"tool_name": "read_file"},
        ]]
        speedup = self.advisor.estimate_speedup(plan)
        # 3 reads in parallel: sequential = 0.9s, parallel = 0.3s -> 3x
        assert speedup == 3.0

    def test_estimate_speedup_sequential(self):
        plan = [
            [{"tool_name": "read_file"}],
            [{"tool_name": "write_file"}],
        ]
        speedup = self.advisor.estimate_speedup(plan)
        # Each batch has one item, no parallelism
        assert speedup == 1.0

    def test_estimate_speedup_empty(self):
        assert self.advisor.estimate_speedup([]) == 1.0

    def test_estimate_speedup_mixed(self):
        plan = [
            [
                {"tool_name": "read_file"},
                {"tool_name": "search_files"},
            ],
            [{"tool_name": "write_file"}],
        ]
        speedup = self.advisor.estimate_speedup(plan)
        # Batch 1: seq=0.3+0.8=1.1, par=0.8; Batch 2: 0.4
        # Total seq=1.5, par=1.2 -> 1.25
        assert speedup == pytest.approx(1.25, abs=0.01)


# -----------------------------------------------------------------------
# Integration
# -----------------------------------------------------------------------

class TestIntegration:

    def test_full_workflow(self):
        """End-to-end: predict, cache, batch."""
        # 1. History of tool calls
        history = [
            _rec("search_files", args={"pattern": "TODO"}, result="/a.py\n/b.py", ts=1.0),
        ]

        # 2. Predict next calls
        predictor = PrefetchPredictor()
        predictions = predictor.predict_next_tools(history)
        assert len(predictions) > 0

        # 3. Check if we should batch the predictions
        advisor = BatchingAdvisor()
        pred_dicts = [{"tool_name": p["tool_name"], "args": p["args"]} for p in predictions]
        # Add another read to make it batchable
        pred_dicts.append({"tool_name": "read_file", "args": {"path": "/c.py"}})

        if advisor.should_batch(pred_dicts):
            batches = advisor.suggest_batch(pred_dicts)
            speedup = advisor.estimate_speedup(batches)
            assert speedup >= 1.0

        # 4. Cache results
        cache = PrefetchCache()
        for p in predictions:
            key = PrefetchCache.make_key(p["tool_name"], p["args"])
            cache.put(key, "prefetched_result")

        # Verify cache works
        key = PrefetchCache.make_key(predictions[0]["tool_name"], predictions[0]["args"])
        assert cache.get(key) == "prefetched_result"

    def test_analyzer_graph_order_matches_groups(self):
        """Verify graph execution order matches find_parallelizable_groups."""
        analyzer = DependencyAnalyzer()
        calls = [
            _rec("read_file", args={"path": "/a.py"}, call_id="r1", ts=1.0),
            _rec("read_file", args={"path": "/b.py"}, call_id="r2", ts=2.0),
            _rec("write_file", args={"path": "/a.py"}, call_id="w1", ts=3.0),
            _rec("terminal", args={"command": "test"}, call_id="t1", ts=4.0),
        ]
        graph = analyzer.analyze_dependencies(calls)
        order = graph.get_execution_order()
        groups = analyzer.find_parallelizable_groups(calls)

        # Both should produce the same grouping structure
        assert len(order) == len(groups)
        for o_group, g_group in zip(order, groups):
            assert set(r.call_id for r in o_group) == set(r.call_id for r in g_group)
