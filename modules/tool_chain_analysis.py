"""
Causal Inference on Tool Chains (#13)

Analyzes tool call sequences to find patterns, detect loops,
and suggest optimal orderings based on historical success data.
"""

import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from pathlib import Path


@dataclass
class Pattern:
    """A tool sequence pattern with associated metrics."""
    sequence: Tuple[str, ...]
    occurrences: int
    success_rate: float
    avg_duration: float
    description: str = ""

    def __post_init__(self):
        if not self.description:
            seq_str = " -> ".join(self.sequence)
            if self.success_rate < 0.3:
                self.description = f"Low success pattern ({self.success_rate:.0%}): {seq_str}"
            elif self.success_rate > 0.8:
                self.description = f"High success pattern ({self.success_rate:.0%}): {seq_str}"
            else:
                self.description = f"Mixed pattern ({self.success_rate:.0%}): {seq_str}"


@dataclass
class ChainRecord:
    """A recorded tool chain execution."""
    session_id: str
    tool_sequence: List[str]
    outcome: str  # "success" or "failure"
    duration: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ToolChainLogger:
    """Logs tool chain executions to SQLite for later analysis."""

    def __init__(self, db_path: str = "tool_chains.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_chains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_sequence TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    duration REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON tool_chains(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome
                ON tool_chains(outcome)
            """)

    def log_chain(self, session_id: str, tool_sequence: List[str],
                  outcome: str, duration: float) -> int:
        """Log a tool chain execution. Returns the row ID."""
        if outcome not in ("success", "failure"):
            raise ValueError(f"outcome must be 'success' or 'failure', got '{outcome}'")
        if not tool_sequence:
            raise ValueError("tool_sequence cannot be empty")

        timestamp = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO tool_chains (session_id, tool_sequence, outcome, duration, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, json.dumps(tool_sequence), outcome, duration, timestamp)
            )
            return cursor.lastrowid

    def get_chains(self, min_timestamp: float = 0,
                   outcome: Optional[str] = None) -> List[ChainRecord]:
        """Retrieve logged chains with optional filters."""
        query = "SELECT session_id, tool_sequence, outcome, duration, timestamp FROM tool_chains WHERE timestamp >= ?"
        params: list = [min_timestamp]
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        query += " ORDER BY timestamp"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            ChainRecord(
                session_id=r[0],
                tool_sequence=json.loads(r[1]),
                outcome=r[2],
                duration=r[3],
                timestamp=r[4]
            )
            for r in rows
        ]

    def count_chains(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM tool_chains").fetchone()[0]


class CausalAnalyzer:
    """Builds DAGs from tool chains and analyzes causal relationships."""

    def __init__(self, logger: ToolChainLogger):
        self.logger = logger
        self._dag: Dict[str, Dict[str, int]] = {}  # from_tool -> {to_tool: count}
        self._success_dag: Dict[str, Dict[str, int]] = {}
        self._subsequence_stats: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "total_duration": 0.0}
        )

    def build_dag(self, min_chains: int = 50) -> Dict[str, Dict[str, int]]:
        """Build a tool sequence DAG from logged data.

        Returns the DAG as adjacency dict, or empty dict if insufficient data.
        """
        chains = self.logger.get_chains()
        if len(chains) < min_chains:
            return {}

        self._dag = defaultdict(lambda: defaultdict(int))
        self._success_dag = defaultdict(lambda: defaultdict(int))
        self._subsequence_stats = defaultdict(
            lambda: {"success": 0, "failure": 0, "total_duration": 0.0}
        )

        for chain in chains:
            seq = chain.tool_sequence
            # Build transition counts
            for i in range(len(seq) - 1):
                self._dag[seq[i]][seq[i + 1]] += 1
                if chain.outcome == "success":
                    self._success_dag[seq[i]][seq[i + 1]] += 1

            # Build subsequence stats for all subsequences of length 2-4
            for length in range(2, min(5, len(seq) + 1)):
                for start in range(len(seq) - length + 1):
                    subseq = tuple(seq[start:start + length])
                    stats = self._subsequence_stats[subseq]
                    if chain.outcome == "success":
                        stats["success"] += 1
                    else:
                        stats["failure"] += 1
                    stats["total_duration"] += chain.duration

        return dict(self._dag)

    def compute_success_rate(self, tool_sequence: List[str]) -> float:
        """Compute historical success rate for a given tool sequence."""
        key = tuple(tool_sequence)
        if key in self._subsequence_stats:
            stats = self._subsequence_stats[key]
            total = stats["success"] + stats["failure"]
            if total > 0:
                return stats["success"] / total
        # Fall back to checking individual transitions
        if len(tool_sequence) < 2:
            return 0.5  # No data, neutral
        success_transitions = 0
        total_transitions = 0
        for i in range(len(tool_sequence) - 1):
            a, b = tool_sequence[i], tool_sequence[i + 1]
            if a in self._dag and b in self._dag[a]:
                total_transitions += self._dag[a][b]
                if a in self._success_dag and b in self._success_dag[a]:
                    success_transitions += self._success_dag[a][b]
        if total_transitions == 0:
            return 0.5
        return success_transitions / total_transitions

    def find_bad_patterns(self, max_results: int = 10) -> List[Pattern]:
        """Find tool sequences with low success rates."""
        patterns = []
        for seq, stats in self._subsequence_stats.items():
            total = stats["success"] + stats["failure"]
            if total < 3:  # Need minimum observations
                continue
            rate = stats["success"] / total
            if rate < 0.4:
                avg_dur = stats["total_duration"] / total
                patterns.append(Pattern(
                    sequence=seq,
                    occurrences=total,
                    success_rate=rate,
                    avg_duration=avg_dur
                ))
        patterns.sort(key=lambda p: (p.success_rate, -p.occurrences))
        return patterns[:max_results]

    def find_optimal_orderings(self, task_type: Optional[str] = None) -> List[Pattern]:  # noqa: ARG002
        """Find tool sequences with high success rates.

        Args:
            task_type: Reserved for future use (filter by task category). Currently unused.
        """
        patterns = []
        for seq, stats in self._subsequence_stats.items():
            total = stats["success"] + stats["failure"]
            if total < 3:
                continue
            rate = stats["success"] / total
            if rate > 0.7:
                avg_dur = stats["total_duration"] / total
                patterns.append(Pattern(
                    sequence=seq,
                    occurrences=total,
                    success_rate=rate,
                    avg_duration=avg_dur
                ))
        patterns.sort(key=lambda p: (-p.success_rate, -p.occurrences))
        return patterns[:10]

    def detect_loop(self, current_chain: List[str]) -> bool:
        """Check if the current chain matches a known-bad loop pattern."""
        if len(current_chain) < 3:
            return False
        # Check against bad patterns
        bad_patterns = self.find_bad_patterns()
        for pattern in bad_patterns:
            if pattern.success_rate < 0.2:
                seq = list(pattern.sequence)
                # Check if current chain ends with this pattern
                if len(current_chain) >= len(seq):
                    if current_chain[-len(seq):] == seq:
                        return True
        # Also use heuristic loop detection
        detector = LoopDetector()
        return detector.is_stuck(current_chain, [])


class LoopDetector:
    """Detects when an agent is stuck in a loop and suggests escapes."""

    def __init__(self, repeat_threshold: int = 3):
        self.repeat_threshold = repeat_threshold

    def is_stuck(self, recent_tools: List[str], recent_errors: List[str]) -> bool:
        """Determine if the agent is stuck in a loop.

        Heuristics:
        - 3+ identical consecutive tool calls
        - error-tool-error repeating pattern
        - Same tool called 5+ times in last 10 calls
        """
        if not recent_tools:
            return False

        # Check for consecutive identical calls
        if len(recent_tools) >= self.repeat_threshold:
            last_n = recent_tools[-self.repeat_threshold:]
            if len(set(last_n)) == 1:
                return True

        # Check for error-tool-error pattern
        if len(recent_errors) >= 2 and len(recent_tools) >= 2:
            # If we have errors alternating with tool calls
            error_indices = set()
            for i, tool in enumerate(recent_tools):
                if i < len(recent_errors) and recent_errors[i]:
                    error_indices.add(i)
            # Pattern: at least 2 errors in last 4 calls
            recent_window = min(4, len(recent_tools))
            recent_error_count = sum(
                1 for i in range(len(recent_tools) - recent_window, len(recent_tools))
                if i in error_indices
            )
            if recent_error_count >= 2:
                return True

        # Check for same tool called too many times in recent window
        window = recent_tools[-10:] if len(recent_tools) > 10 else recent_tools
        counts = Counter(window)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 5:
            return True

        # Check for repeating subsequence (e.g., A-B-A-B-A-B)
        if len(recent_tools) >= 6:
            for sublen in range(2, 4):
                pattern = recent_tools[-sublen:]
                repeats = 0
                for offset in range(0, len(recent_tools) - sublen + 1, sublen):
                    chunk = recent_tools[-(offset + sublen):len(recent_tools) - offset] if offset > 0 else recent_tools[-sublen:]
                    start = len(recent_tools) - offset - sublen
                    end = len(recent_tools) - offset
                    if start >= 0 and recent_tools[start:end] == pattern:
                        repeats += 1
                if repeats >= 3:
                    return True

        return False

    def suggest_escape(self, current_chain: List[str]) -> str:
        """Suggest an escape strategy for a stuck chain."""
        if not current_chain:
            return "No chain data available. Try starting with a different approach."

        # Analyze what's repeating
        last_tools = current_chain[-6:] if len(current_chain) > 6 else current_chain
        counts = Counter(last_tools)
        most_repeated = counts.most_common(1)[0]

        suggestions = []

        # If same tool is being called repeatedly
        if most_repeated[1] >= 3:
            tool = most_repeated[0]
            suggestions.append(
                f"Tool '{tool}' has been called {most_repeated[1]} times recently. "
                f"Step back and try a different approach."
            )
            # Suggest alternatives based on common patterns
            if "search" in tool.lower() or "read" in tool.lower():
                suggestions.append(
                    "Consider: Instead of searching/reading repeatedly, "
                    "try writing the solution with what you already know."
                )
            elif "write" in tool.lower() or "edit" in tool.lower():
                suggestions.append(
                    "Consider: Instead of editing repeatedly, "
                    "read the current state first, plan the full change, then apply it once."
                )
            elif "terminal" in tool.lower() or "bash" in tool.lower():
                suggestions.append(
                    "Consider: If terminal commands keep failing, "
                    "check prerequisites or try a completely different approach."
                )

        # If there's an alternating pattern
        if len(last_tools) >= 4:
            pairs = [(last_tools[i], last_tools[i + 1]) for i in range(len(last_tools) - 1)]
            pair_counts = Counter(pairs)
            if pair_counts.most_common(1)[0][1] >= 2:
                pair = pair_counts.most_common(1)[0][0]
                suggestions.append(
                    f"Detected alternating pattern: {pair[0]} <-> {pair[1]}. "
                    f"Break the cycle by using a completely different tool or approach."
                )

        if not suggestions:
            suggestions.append(
                "The current approach seems stuck. "
                "Try: 1) Re-read the requirements, 2) Use a different tool, "
                "3) Simplify the problem."
            )

        return " | ".join(suggestions)
