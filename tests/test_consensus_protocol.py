"""Tests for consensus_protocol.py — Multi-Agent Consensus Protocol (#15)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from consensus_protocol import (
    AgentResult,
    Conflict,
    ConsensusProtocol,
    ConsensusVerifier,
    DiffMerger,
)


def make_result(agent_id, output="done", confidence=0.8,
                files=None, duration=10.0):
    return AgentResult(
        agent_id=agent_id,
        output=output,
        confidence=confidence,
        files_changed=files or {},
        duration=duration
    )


class TestAgentResult:
    def test_valid(self):
        r = make_result("a1")
        assert r.agent_id == "a1"
        assert r.confidence == 0.8

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="Confidence"):
            AgentResult("a1", "out", 1.5, {}, 1.0)


class TestConsensusProtocol:
    def test_add_result(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1"))
        assert len(cp.results) == 1

    def test_duplicate_agent_raises(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1"))
        with pytest.raises(ValueError, match="already"):
            cp.add_result(make_result("a1"))

    def test_agreement_single_agent(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1"))
        assert cp.compute_agreement() == 1.0

    def test_jaccard_agreement_same_files(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", files={"f1": "x", "f2": "y"}))
        cp.add_result(make_result("a2", files={"f1": "x", "f2": "y"}))
        assert cp.compute_agreement("jaccard") == 1.0

    def test_jaccard_agreement_different_files(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", files={"f1": "x"}))
        cp.add_result(make_result("a2", files={"f2": "y"}))
        assert cp.compute_agreement("jaccard") == 0.0

    def test_jaccard_agreement_partial(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", files={"f1": "x", "f2": "y"}))
        cp.add_result(make_result("a2", files={"f1": "x", "f3": "z"}))
        # Jaccard: 1 / 3 = 0.333
        score = cp.compute_agreement("jaccard")
        assert abs(score - 1 / 3) < 0.01

    def test_output_agreement(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", output="hello world"))
        cp.add_result(make_result("a2", output="hello world"))
        assert cp.compute_agreement("output") == 1.0

    def test_majority_vote_output(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", output="answer A"))
        cp.add_result(make_result("a2", output="answer A"))
        cp.add_result(make_result("a3", output="answer B"))
        result = cp.vote("majority")
        assert result.output == "answer A"
        assert result.strategy_used == "majority"

    def test_majority_vote_files(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", files={"f.py": "v1"}))
        cp.add_result(make_result("a2", files={"f.py": "v1"}))
        cp.add_result(make_result("a3", files={"f.py": "v2"}, confidence=0.9))
        result = cp.vote("majority")
        assert result.files_changed["f.py"] == "v1"

    def test_confidence_weighted(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", output="low", confidence=0.3,
                                  files={"f.py": "low_v"}))
        cp.add_result(make_result("a2", output="high", confidence=0.95,
                                  files={"f.py": "high_v"}))
        result = cp.vote("confidence_weighted")
        assert result.output == "high"
        assert result.files_changed["f.py"] == "high_v"

    def test_union_merge(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1", output="part1", files={"f1.py": "c1"}))
        cp.add_result(make_result("a2", output="part2", files={"f2.py": "c2"}))
        result = cp.vote("union_merge")
        assert "f1.py" in result.files_changed
        assert "f2.py" in result.files_changed
        assert "part1" in result.output
        assert "part2" in result.output

    def test_vote_empty_raises(self):
        cp = ConsensusProtocol()
        with pytest.raises(ValueError, match="No results"):
            cp.vote()

    def test_unknown_strategy_raises(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1"))
        with pytest.raises(ValueError, match="Unknown strategy"):
            cp.vote("quantum")

    def test_clear(self):
        cp = ConsensusProtocol()
        cp.add_result(make_result("a1"))
        cp.clear()
        assert len(cp.results) == 0


class TestDiffMerger:
    def test_no_conflicts(self):
        results = [
            make_result("a1", files={"f.py": "same"}),
            make_result("a2", files={"f.py": "same"}),
        ]
        merger = DiffMerger()
        merged = merger.merge_file_changes(results)
        assert len(merged) == 1
        assert merged[0].had_conflicts is False
        assert len(merged[0].source_agents) == 2

    def test_with_conflicts(self):
        results = [
            make_result("a1", files={"f.py": "version_a"}, confidence=0.9),
            make_result("a2", files={"f.py": "version_b"}, confidence=0.7),
        ]
        merger = DiffMerger()
        merged = merger.merge_file_changes(results)
        assert len(merged) == 1
        assert merged[0].had_conflicts is True
        assert merged[0].content == "version_a"  # Higher confidence

    def test_detect_conflicts(self):
        results = [
            make_result("a1", files={"f.py": "v1"}),
            make_result("a2", files={"f.py": "v2"}),
        ]
        merger = DiffMerger()
        conflicts = merger.detect_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].filename == "f.py"

    def test_no_conflicts_detected(self):
        results = [
            make_result("a1", files={"f.py": "same"}),
            make_result("a2", files={"f.py": "same"}),
        ]
        merger = DiffMerger()
        assert len(merger.detect_conflicts(results)) == 0

    def test_resolve_conflict_longest(self):
        conflict = Conflict(
            filename="f.py",
            agents=["a1", "a2"],
            contents={"a1": "short", "a2": "much longer content here"},
        )
        merger = DiffMerger()
        resolution = merger.resolve_conflict(conflict, strategy="longest")
        assert resolution.resolved_content == "much longer content here"
        assert resolution.agents_favored == ["a2"]

    def test_resolve_conflict_confidence(self):
        conflict = Conflict(
            filename="f.py",
            agents=["a1", "a2"],
            contents={"a1": "content_a", "a2": "content_b"},
        )
        merger = DiffMerger()
        resolution = merger.resolve_conflict(
            conflict, strategy="confidence",
            agent_confidences={"a1": 0.9, "a2": 0.5}
        )
        assert resolution.resolved_content == "content_a"

    def test_resolve_conflict_first(self):
        conflict = Conflict(
            filename="f.py",
            agents=["a1", "a2"],
            contents={"a1": "first", "a2": "second"},
        )
        merger = DiffMerger()
        resolution = merger.resolve_conflict(conflict, strategy="first")
        assert resolution.resolved_content == "first"


class TestConsensusVerifier:
    def test_preservation_all_included(self):
        merged = "This is line one from agent 1\nThis is line two from agent 2"
        results = [
            make_result("a1", output="This is line one from agent 1"),
            make_result("a2", output="This is line two from agent 2"),
        ]
        verifier = ConsensusVerifier()
        assert verifier.verify_preservation(None, merged, results) is True

    def test_preservation_empty_agents(self):
        verifier = ConsensusVerifier()
        assert verifier.verify_preservation(None, "anything", []) is True

    def test_file_preservation(self):
        merged_files = {"f.py": "combined content here"}
        results = [
            make_result("a1", files={"f.py": "combined content here"}),
        ]
        verifier = ConsensusVerifier()
        file_results = verifier.verify_file_preservation(merged_files, results)
        assert file_results["a1:f.py"] is True

    def test_file_not_preserved(self):
        merged_files = {"other.py": "something"}
        results = [
            make_result("a1", files={"f.py": "my content"}),
        ]
        verifier = ConsensusVerifier()
        file_results = verifier.verify_file_preservation(merged_files, results)
        assert file_results["a1:f.py"] is False
