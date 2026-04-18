"""
Multi-Agent Consensus Protocol (#15)

Protocol for combining results from multiple agents, detecting conflicts,
and merging file changes with configurable strategies.
"""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ConflictStrategy(str, Enum):
    MAJORITY_VOTE = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    UNION_MERGE = "union_merge"


@dataclass
class AgentResult:
    """Result from a single agent's execution."""
    agent_id: str
    output: str
    confidence: float  # 0.0 - 1.0
    files_changed: Dict[str, str]  # filename -> new content
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass
class MergedChange:
    """A merged file change from multiple agents."""
    filename: str
    content: str
    source_agents: List[str]
    confidence: float
    had_conflicts: bool = False


@dataclass
class Conflict:
    """A conflict between agent changes."""
    filename: str
    agents: List[str]
    contents: Dict[str, str]  # agent_id -> content
    region: Optional[str] = None  # description of conflicting region

    def __str__(self):
        return (f"Conflict in '{self.filename}' between agents "
                f"{', '.join(self.agents)}: {self.region or 'full file'}")


@dataclass
class Resolution:
    """Resolution of a conflict."""
    conflict: Conflict
    resolved_content: str
    strategy_used: str
    agents_favored: List[str]


@dataclass
class MergedResult:
    """Final merged result from consensus."""
    output: str
    confidence: float
    files_changed: Dict[str, str]
    contributing_agents: List[str]
    agreement_score: float
    strategy_used: str
    conflicts: List[Conflict] = field(default_factory=list)
    resolutions: List[Resolution] = field(default_factory=list)


class ConsensusProtocol:
    """Combines results from multiple agents using configurable strategies."""

    def __init__(self):
        self._results: List[AgentResult] = []

    def add_result(self, agent_result: AgentResult):
        """Add an agent's result to the consensus pool."""
        # Check for duplicate agent_id
        for existing in self._results:
            if existing.agent_id == agent_result.agent_id:
                raise ValueError(f"Agent '{agent_result.agent_id}' already submitted a result.")
        self._results.append(agent_result)

    def clear(self):
        """Clear all results."""
        self._results.clear()

    @property
    def results(self) -> List[AgentResult]:
        return list(self._results)

    def compute_agreement(self, metric: str = "jaccard") -> float:
        """Compute pairwise agreement between agents.

        Metrics:
        - jaccard: Jaccard similarity of files changed
        - output: Text similarity of outputs
        - files_content: Content similarity of changed files
        """
        if len(self._results) < 2:
            return 1.0

        if metric == "jaccard":
            return self._jaccard_agreement()
        elif metric == "output":
            return self._output_agreement()
        elif metric == "files_content":
            return self._content_agreement()
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jaccard', 'output', or 'files_content'.")

    def _jaccard_agreement(self) -> float:
        """Average pairwise Jaccard similarity of files changed."""
        scores = []
        for i in range(len(self._results)):
            for j in range(i + 1, len(self._results)):
                files_a = set(self._results[i].files_changed.keys())
                files_b = set(self._results[j].files_changed.keys())
                if not files_a and not files_b:
                    scores.append(1.0)
                elif not files_a or not files_b:
                    scores.append(0.0)
                else:
                    intersection = files_a & files_b
                    union = files_a | files_b
                    scores.append(len(intersection) / len(union))
        return sum(scores) / len(scores) if scores else 1.0

    def _output_agreement(self) -> float:
        """Average pairwise text similarity of outputs."""
        scores = []
        for i in range(len(self._results)):
            for j in range(i + 1, len(self._results)):
                ratio = SequenceMatcher(
                    None,
                    self._results[i].output,
                    self._results[j].output
                ).ratio()
                scores.append(ratio)
        return sum(scores) / len(scores) if scores else 1.0

    def _content_agreement(self) -> float:
        """Average pairwise content similarity of changed files."""
        scores = []
        for i in range(len(self._results)):
            for j in range(i + 1, len(self._results)):
                common_files = (set(self._results[i].files_changed.keys()) &
                                set(self._results[j].files_changed.keys()))
                if not common_files:
                    scores.append(0.0)
                    continue
                file_scores = []
                for f in common_files:
                    ratio = SequenceMatcher(
                        None,
                        self._results[i].files_changed[f],
                        self._results[j].files_changed[f]
                    ).ratio()
                    file_scores.append(ratio)
                scores.append(sum(file_scores) / len(file_scores))
        return sum(scores) / len(scores) if scores else 1.0

    def vote(self, strategy: str = "majority") -> MergedResult:
        """Merge results using the specified strategy."""
        if not self._results:
            raise ValueError("No results to vote on.")

        if strategy == "majority":
            return self._majority_vote()
        elif strategy == "confidence_weighted":
            return self._confidence_weighted()
        elif strategy == "union_merge":
            return self._union_merge()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _majority_vote(self) -> MergedResult:
        """Select the most common output/file version."""
        # Vote on output
        output_counts: Counter = Counter()
        output_by_hash: Dict[str, str] = {}
        for r in self._results:
            h = hashlib.md5(r.output.encode()).hexdigest()
            output_counts[h] += 1
            output_by_hash[h] = r.output
        best_output_hash = output_counts.most_common(1)[0][0]
        best_output = output_by_hash[best_output_hash]

        # Vote on each file
        all_files = set()
        for r in self._results:
            all_files.update(r.files_changed.keys())

        merged_files: Dict[str, str] = {}
        conflicts = []
        resolutions = []

        for filename in all_files:
            # Get all versions
            versions: Dict[str, str] = {}
            for r in self._results:
                if filename in r.files_changed:
                    versions[r.agent_id] = r.files_changed[filename]

            if len(versions) <= 1:
                # Only one agent changed this file
                agent_id, content = next(iter(versions.items()))
                merged_files[filename] = content
            else:
                # Vote on content
                content_counts: Counter = Counter()
                content_by_hash: Dict[str, Tuple[str, str]] = {}
                for agent_id, content in versions.items():
                    h = hashlib.md5(content.encode()).hexdigest()
                    content_counts[h] += 1
                    content_by_hash[h] = (content, agent_id)

                best_hash = content_counts.most_common(1)[0]
                if best_hash[1] > 1:
                    # Majority exists
                    merged_files[filename] = content_by_hash[best_hash[0]][0]
                else:
                    # No majority — conflict
                    conflict = Conflict(
                        filename=filename,
                        agents=list(versions.keys()),
                        contents=versions,
                        region="full file (no majority)"
                    )
                    conflicts.append(conflict)
                    # Resolve by picking highest confidence agent
                    best_agent = max(
                        versions.keys(),
                        key=lambda a: next(r.confidence for r in self._results if r.agent_id == a)
                    )
                    merged_files[filename] = versions[best_agent]
                    resolutions.append(Resolution(
                        conflict=conflict,
                        resolved_content=versions[best_agent],
                        strategy_used="majority_fallback_confidence",
                        agents_favored=[best_agent]
                    ))

        agreement = self.compute_agreement()
        avg_confidence = sum(r.confidence for r in self._results) / len(self._results)

        return MergedResult(
            output=best_output,
            confidence=avg_confidence,
            files_changed=merged_files,
            contributing_agents=[r.agent_id for r in self._results],
            agreement_score=agreement,
            strategy_used="majority",
            conflicts=conflicts,
            resolutions=resolutions
        )

    def _confidence_weighted(self) -> MergedResult:
        """Select output from highest-confidence agent, merge files by confidence."""
        best_agent = max(self._results, key=lambda r: r.confidence)

        # For files, prefer highest confidence agent's version
        all_files = set()
        for r in self._results:
            all_files.update(r.files_changed.keys())

        merged_files: Dict[str, str] = {}
        conflicts = []

        for filename in all_files:
            versions = {r.agent_id: r.files_changed[filename]
                        for r in self._results if filename in r.files_changed}

            if len(set(versions.values())) <= 1:
                merged_files[filename] = next(iter(versions.values()))
            else:
                # Pick version from highest confidence agent that changed this file
                best = max(
                    ((r.agent_id, r.confidence) for r in self._results
                     if filename in r.files_changed),
                    key=lambda x: x[1]
                )
                merged_files[filename] = versions[best[0]]
                if len(set(versions.values())) > 1:
                    conflicts.append(Conflict(
                        filename=filename,
                        agents=list(versions.keys()),
                        contents=versions,
                        region="resolved by confidence"
                    ))

        agreement = self.compute_agreement()

        return MergedResult(
            output=best_agent.output,
            confidence=best_agent.confidence,
            files_changed=merged_files,
            contributing_agents=[r.agent_id for r in self._results],
            agreement_score=agreement,
            strategy_used="confidence_weighted",
            conflicts=conflicts
        )

    def _union_merge(self) -> MergedResult:
        """Include all file changes (union). Conflicts resolved by confidence."""
        # Combine outputs
        outputs = [f"[{r.agent_id}]: {r.output}" for r in self._results]
        combined_output = "\n---\n".join(outputs)

        all_files = set()
        for r in self._results:
            all_files.update(r.files_changed.keys())

        merged_files: Dict[str, str] = {}
        conflicts = []

        for filename in all_files:
            versions = {r.agent_id: r.files_changed[filename]
                        for r in self._results if filename in r.files_changed}

            unique_contents = set(versions.values())
            if len(unique_contents) == 1:
                merged_files[filename] = next(iter(versions.values()))
            else:
                # Try to merge, fall back to highest confidence
                best = max(
                    ((r.agent_id, r.confidence) for r in self._results
                     if filename in r.files_changed),
                    key=lambda x: x[1]
                )
                merged_files[filename] = versions[best[0]]
                conflicts.append(Conflict(
                    filename=filename,
                    agents=list(versions.keys()),
                    contents=versions,
                    region="union merge conflict"
                ))

        agreement = self.compute_agreement()
        avg_confidence = sum(r.confidence for r in self._results) / len(self._results)

        return MergedResult(
            output=combined_output,
            confidence=avg_confidence,
            files_changed=merged_files,
            contributing_agents=[r.agent_id for r in self._results],
            agreement_score=agreement,
            strategy_used="union_merge",
            conflicts=conflicts
        )


class DiffMerger:
    """Merges file changes from multiple agents."""

    def merge_file_changes(self, agent_results: List[AgentResult]) -> List[MergedChange]:
        """Merge file changes across all agent results."""
        all_files: Dict[str, Dict[str, str]] = defaultdict(dict)  # file -> {agent: content}

        for result in agent_results:
            for filename, content in result.files_changed.items():
                all_files[filename][result.agent_id] = content

        merged = []
        for filename, versions in all_files.items():
            unique_contents = set(versions.values())

            if len(unique_contents) == 1:
                # All agents agree
                content = next(iter(versions.values()))
                agents = list(versions.keys())
                avg_conf = sum(
                    r.confidence for r in agent_results
                    if r.agent_id in agents
                ) / len(agents)
                merged.append(MergedChange(
                    filename=filename,
                    content=content,
                    source_agents=agents,
                    confidence=avg_conf,
                    had_conflicts=False
                ))
            else:
                # Conflict — pick highest confidence
                confidences = {
                    r.agent_id: r.confidence
                    for r in agent_results
                    if r.agent_id in versions
                }
                best_agent = max(confidences, key=confidences.get)
                merged.append(MergedChange(
                    filename=filename,
                    content=versions[best_agent],
                    source_agents=[best_agent],
                    confidence=confidences[best_agent],
                    had_conflicts=True
                ))

        return merged

    def detect_conflicts(self, agent_results: List[AgentResult]) -> List[Conflict]:
        """Detect conflicting changes across agents."""
        all_files: Dict[str, Dict[str, str]] = defaultdict(dict)

        for result in agent_results:
            for filename, content in result.files_changed.items():
                all_files[filename][result.agent_id] = content

        conflicts = []
        for filename, versions in all_files.items():
            unique_contents = set(versions.values())
            if len(unique_contents) > 1:
                conflicts.append(Conflict(
                    filename=filename,
                    agents=list(versions.keys()),
                    contents=versions,
                    region=self._describe_conflict_region(versions)
                ))

        return conflicts

    def _describe_conflict_region(self, versions: Dict[str, str]) -> str:
        """Describe where versions differ."""
        contents = list(versions.values())
        if len(contents) < 2:
            return "no conflict"

        lines_a = contents[0].splitlines()
        lines_b = contents[1].splitlines()

        sm = SequenceMatcher(None, lines_a, lines_b)
        differing_lines = []
        for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
            if tag != "equal":
                differing_lines.append(f"lines {i1 + 1}-{i2}")

        if differing_lines:
            return f"differs at {', '.join(differing_lines)}"
        return "identical content"

    def resolve_conflict(self, conflict: Conflict,
                         strategy: str = "longest",
                         agent_confidences: Optional[Dict[str, float]] = None) -> Resolution:
        """Resolve a single conflict using the specified strategy."""
        if strategy == "longest":
            # Pick the longest content (most work done)
            best = max(conflict.contents.items(), key=lambda x: len(x[1]))
            return Resolution(
                conflict=conflict,
                resolved_content=best[1],
                strategy_used="longest",
                agents_favored=[best[0]]
            )
        elif strategy == "confidence" and agent_confidences:
            # Pick highest confidence agent
            best_agent = max(
                (a for a in conflict.agents if a in agent_confidences),
                key=lambda a: agent_confidences[a]
            )
            return Resolution(
                conflict=conflict,
                resolved_content=conflict.contents[best_agent],
                strategy_used="confidence",
                agents_favored=[best_agent]
            )
        elif strategy == "first":
            first_agent = conflict.agents[0]
            return Resolution(
                conflict=conflict,
                resolved_content=conflict.contents[first_agent],
                strategy_used="first",
                agents_favored=[first_agent]
            )
        else:
            # Default: pick first
            first_agent = conflict.agents[0]
            return Resolution(
                conflict=conflict,
                resolved_content=conflict.contents[first_agent],
                strategy_used="default",
                agents_favored=[first_agent]
            )


class ConsensusVerifier:
    """Verifies that merged results preserve agent contributions."""

    def verify_preservation(self, original: Optional[str],
                            merged: str,
                            agent_results: List[AgentResult]) -> bool:
        """Verify each agent's correct contributions are preserved.

        Checks that key content from each agent's output appears in the merge,
        weighted by confidence.
        """
        if not agent_results:
            return True

        # Extract significant tokens from each agent's output
        preserved_count = 0
        total_count = 0

        for result in agent_results:
            # Get significant lines from this agent's output
            agent_lines = set(result.output.strip().splitlines())
            significant_lines = {
                line.strip() for line in agent_lines
                if len(line.strip()) > 10  # Skip trivial lines
            }

            if not significant_lines:
                preserved_count += 1
                total_count += 1
                continue

            # Check what fraction of significant lines appear in merged
            found = sum(1 for line in significant_lines if line in merged)
            fraction = found / len(significant_lines) if significant_lines else 1.0

            # Weight by confidence — high confidence results should be more preserved
            if fraction >= (1.0 - result.confidence) * 0.5:
                preserved_count += 1
            total_count += 1

        # At least majority of agents' contributions should be preserved
        return preserved_count / total_count >= 0.5 if total_count > 0 else True

    def verify_file_preservation(self, merged_files: Dict[str, str],
                                 agent_results: List[AgentResult]) -> Dict[str, bool]:
        """Verify file-level preservation of each agent's changes."""
        results = {}
        for result in agent_results:
            for filename, content in result.files_changed.items():
                if filename in merged_files:
                    # Check if the merged version contains key elements
                    merged = merged_files[filename]
                    similarity = SequenceMatcher(None, content, merged).ratio()
                    results[f"{result.agent_id}:{filename}"] = similarity > 0.3
                else:
                    results[f"{result.agent_id}:{filename}"] = False
        return results
