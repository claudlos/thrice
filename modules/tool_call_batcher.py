"""
Intelligent Tool Call Batching (#8)

Detects independent tool calls and suggests parallel execution.
Analyzes dependencies between tool calls, predicts next likely calls,
and advises on optimal batching strategies for improved throughput.
"""

import time
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """A single recorded tool call with dependency metadata."""
    tool_name: str
    args: Dict
    timestamp: float = 0.0
    result: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    call_id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = uuid.uuid4().hex[:12]
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def get_target_path(self) -> Optional[str]:
        """Extract the file path this call targets, if any."""
        return (
            self.args.get("path")
            or self.args.get("file")
            or self.args.get("filename")
        )


class ToolCategory(Enum):
    """Broad categories for dependency analysis."""
    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    EXECUTE = "execute"
    OTHER = "other"


_TOOL_CATEGORIES: Dict[str, ToolCategory] = {
    "read_file": ToolCategory.READ,
    "mcp_read_file": ToolCategory.READ,
    "search_files": ToolCategory.SEARCH,
    "mcp_search_files": ToolCategory.SEARCH,
    "write_file": ToolCategory.WRITE,
    "mcp_write_file": ToolCategory.WRITE,
    "patch": ToolCategory.WRITE,
    "mcp_patch": ToolCategory.WRITE,
    "terminal": ToolCategory.EXECUTE,
    "mcp_terminal": ToolCategory.EXECUTE,
}


def _categorize(tool_name: str) -> ToolCategory:
    """Return the category of a tool by name."""
    return _TOOL_CATEGORIES.get(tool_name, ToolCategory.OTHER)


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """DAG of tool call dependencies.

    Nodes are ToolCallRecords keyed by call_id.
    Edges point from a dependency to its dependent (must-run-before).
    """

    def __init__(self):
        self.nodes: Dict[str, ToolCallRecord] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # from -> {to, ...}
        self._reverse: Dict[str, Set[str]] = defaultdict(set)  # to -> {from, ...}

    def add_node(self, call: ToolCallRecord) -> None:
        self.nodes[call.call_id] = call

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add an edge meaning *from_id* must complete before *to_id*."""
        self.edges[from_id].add(to_id)
        self._reverse[to_id].add(from_id)

    def get_dependencies(self, call_id: str) -> Set[str]:
        """Return IDs that *call_id* depends on."""
        return self._reverse.get(call_id, set())

    def get_dependents(self, call_id: str) -> Set[str]:
        """Return IDs that depend on *call_id*."""
        return self.edges.get(call_id, set())

    def get_execution_order(self) -> List[List[ToolCallRecord]]:
        """Topological sort into parallel execution groups.

        Each inner list contains calls that can run simultaneously.
        Groups are ordered so that every dependency of a call in group N
        appears in a group < N.
        """
        if not self.nodes:
            return []

        in_degree: Dict[str, int] = {cid: 0 for cid in self.nodes}
        for src, targets in self.edges.items():
            for tgt in targets:
                if tgt in in_degree:
                    in_degree[tgt] += 1

        groups: List[List[ToolCallRecord]] = []
        remaining = dict(in_degree)

        while remaining:
            # Collect nodes with no unmet dependencies
            ready = [cid for cid, deg in remaining.items() if deg == 0]
            if not ready:
                # Cycle detected – break it by picking the node with lowest degree
                min_deg = min(remaining.values())
                ready = [cid for cid, deg in remaining.items() if deg == min_deg]

            group = [self.nodes[cid] for cid in ready]
            # Sort within group by timestamp for determinism
            group.sort(key=lambda c: c.timestamp)
            groups.append(group)

            for cid in ready:
                for dep in self.edges.get(cid, set()):
                    if dep in remaining:
                        remaining[dep] -= 1
                del remaining[cid]

        return groups

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# DependencyAnalyzer
# ---------------------------------------------------------------------------

class DependencyAnalyzer:
    """Analyzes tool calls to discover dependencies and parallelism."""

    # Tools whose output is purely read-only and never conflicts
    _READ_ONLY_TOOLS: Set[str] = {
        "read_file", "mcp_read_file",
        "search_files", "mcp_search_files",
    }

    _WRITE_TOOLS: Set[str] = {
        "write_file", "mcp_write_file",
        "patch", "mcp_patch",
    }

    _EXEC_TOOLS: Set[str] = {
        "terminal", "mcp_terminal",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_dependencies(self, calls: List[ToolCallRecord]) -> DependencyGraph:
        """Build a full dependency graph from an ordered list of calls."""
        graph = DependencyGraph()
        for call in calls:
            graph.add_node(call)

        # Pairwise analysis: for each later call, check earlier calls
        for i, later in enumerate(calls):
            for j in range(i):
                earlier = calls[j]
                if not self.are_independent(earlier, later):
                    graph.add_edge(earlier.call_id, later.call_id)

        return graph

    def are_independent(self, call_a: ToolCallRecord, call_b: ToolCallRecord) -> bool:
        """Determine whether two calls can safely run in parallel.

        Rules
        -----
        1. Two read_file calls are always independent.
        2. Two search_files calls are always independent.
        3. A write/patch to file X depends on a prior read of file X.
        4. A write/patch to file X depends on a prior write/patch to file X.
        5. Terminal commands *may* depend on prior terminals (conservative).
        6. Different categories targeting different files are independent.
        """
        cat_a = _categorize(call_a.tool_name)
        cat_b = _categorize(call_b.tool_name)

        # Rule 1 & 2: pure reads / searches are independent of each other
        if cat_a in (ToolCategory.READ, ToolCategory.SEARCH) and \
           cat_b in (ToolCategory.READ, ToolCategory.SEARCH):
            return True

        # Rule 5: two executions are conservatively dependent
        if cat_a == ToolCategory.EXECUTE and cat_b == ToolCategory.EXECUTE:
            return False

        # Rule 3 & 4: writes depend on prior reads/writes of the *same* path
        path_a = call_a.get_target_path()
        path_b = call_b.get_target_path()

        if cat_b == ToolCategory.WRITE:
            # Write depends on prior read/write of same file
            if path_a and path_b and self._same_path(path_a, path_b):
                return False

        if cat_a == ToolCategory.WRITE:
            # Anything touching the same file after a write is dependent
            if path_a and path_b and self._same_path(path_a, path_b):
                return False

        # Write after execute could depend if the execute modifies files
        if cat_a == ToolCategory.EXECUTE and cat_b == ToolCategory.WRITE:
            return False
        if cat_a == ToolCategory.WRITE and cat_b == ToolCategory.EXECUTE:
            return False

        # Default: independent if no path overlap
        return True

    def find_parallelizable_groups(
        self, calls: List[ToolCallRecord]
    ) -> List[List[ToolCallRecord]]:
        """Partition *calls* into groups that can run in parallel."""
        graph = self.analyze_dependencies(calls)
        return graph.get_execution_order()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _same_path(a: str, b: str) -> bool:
        """Loose path comparison (handles trailing slashes, etc.)."""
        return a.rstrip("/") == b.rstrip("/")


# ---------------------------------------------------------------------------
# PrefetchPredictor
# ---------------------------------------------------------------------------

class PrefetchPredictor:
    """Predicts likely next tool calls based on historical patterns.

    Built-in pattern rules (inspired by tool_chain_analysis.py):
      * search_files  -> read_file (open top results)
      * read_file     -> read_file (follow imports)
      * write_file    -> terminal  (syntax check / test)
    """

    # (trigger_tool, predicted_tool, base_confidence)
    _STATIC_PATTERNS: List[Tuple[str, str, float]] = [
        ("search_files", "read_file", 0.85),
        ("mcp_search_files", "mcp_read_file", 0.85),
        ("read_file", "read_file", 0.60),
        ("mcp_read_file", "mcp_read_file", 0.60),
        ("write_file", "terminal", 0.70),
        ("mcp_write_file", "mcp_terminal", 0.70),
        ("mcp_patch", "mcp_terminal", 0.65),
        ("patch", "terminal", 0.65),
    ]

    _IMPORT_RE = re.compile(
        r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))"
    )

    def __init__(self, custom_patterns: Optional[List[Tuple[str, str, float]]] = None):
        self.patterns = list(self._STATIC_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        # Learned transition counts: (from_tool -> {to_tool: count})
        self._transition_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(self, history: List[ToolCallRecord]) -> None:
        """Ingest a history to build transition statistics."""
        for i in range(len(history) - 1):
            src = history[i].tool_name
            dst = history[i + 1].tool_name
            self._transition_counts[src][dst] += 1

    def predict_next_tools(
        self, history: List[ToolCallRecord]
    ) -> List[Dict]:
        """Return predicted next tool calls with confidence scores.

        Each prediction is a dict:
            {"tool_name": str, "args": dict, "confidence": float, "reason": str}
        """
        if not history:
            return []

        predictions: List[Dict] = []
        last = history[-1]

        # --- Static pattern matching ---
        for trigger, predicted, base_conf in self.patterns:
            if last.tool_name == trigger:
                pred = self._build_prediction(last, predicted, base_conf)
                if pred:
                    predictions.append(pred)

        # --- Import-following for read_file ---
        if _categorize(last.tool_name) == ToolCategory.READ and last.result:
            import_preds = self._predict_imports(last)
            predictions.extend(import_preds)

        # --- Learned transitions ---
        if last.tool_name in self._transition_counts:
            totals = self._transition_counts[last.tool_name]
            total_count = sum(totals.values())
            for tool, count in totals.items():
                conf = count / total_count
                if conf >= 0.2:
                    predictions.append({
                        "tool_name": tool,
                        "args": {},
                        "confidence": round(min(conf, 0.95), 2),
                        "reason": f"learned transition ({count}/{total_count})",
                    })

        # Deduplicate by tool_name, keep highest confidence
        seen: Dict[str, Dict] = {}
        for p in predictions:
            key = p["tool_name"]
            if key not in seen or p["confidence"] > seen[key]["confidence"]:
                seen[key] = p
        predictions = sorted(seen.values(), key=lambda p: -p["confidence"])

        return predictions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prediction(
        self, trigger: ToolCallRecord, predicted_tool: str, confidence: float
    ) -> Optional[Dict]:
        """Construct a prediction dict from a trigger call."""
        args: Dict = {}
        reason = ""

        cat = _categorize(trigger.tool_name)
        pred_cat = _categorize(predicted_tool)

        if cat == ToolCategory.SEARCH and pred_cat == ToolCategory.READ:
            # After search, likely to read a found file
            reason = "search -> read top result"
            # If the search result is available, extract a path
            if trigger.result:
                paths = self._extract_paths(trigger.result)
                if paths:
                    args = {"path": paths[0]}
                    confidence = min(confidence + 0.05, 0.95)

        elif cat == ToolCategory.READ and pred_cat == ToolCategory.READ:
            reason = "read -> read (follow reference)"

        elif cat == ToolCategory.WRITE and pred_cat == ToolCategory.EXECUTE:
            path = trigger.get_target_path()
            reason = "write -> syntax/test check"
            if path and path.endswith(".py"):
                args = {"command": f"python -m py_compile {path}"}
                confidence = min(confidence + 0.10, 0.95)

        else:
            reason = "pattern match"

        return {
            "tool_name": predicted_tool,
            "args": args,
            "confidence": round(confidence, 2),
            "reason": reason,
        }

    def _predict_imports(self, read_call: ToolCallRecord) -> List[Dict]:
        """Predict read_file calls for imported modules."""
        if not read_call.result:
            return []

        predictions: List[Dict] = []
        matches = self._IMPORT_RE.findall(read_call.result)
        for from_mod, import_mod in matches:
            mod = from_mod or import_mod
            # Convert dotted module to path guess
            path_guess = mod.replace(".", "/") + ".py"
            predictions.append({
                "tool_name": read_call.tool_name,  # same read tool
                "args": {"path": path_guess},
                "confidence": 0.50,
                "reason": f"import follow: {mod}",
            })
        return predictions[:5]  # Cap to avoid noise

    @staticmethod
    def _extract_paths(text: str) -> List[str]:
        """Pull file paths from search result text."""
        paths = []
        for line in text.splitlines():
            line = line.strip()
            # Heuristic: looks like a file path
            if "/" in line and not line.startswith("#"):
                # Take the first whitespace-delimited token that has a slash
                for token in line.split():
                    if "/" in token and ":" not in token:
                        paths.append(token)
                        break
        return paths[:5]


# ---------------------------------------------------------------------------
# PrefetchCache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    value: str
    created_at: float
    ttl: float

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class PrefetchCache:
    """Cache for prefetched tool results with TTL expiry and hit/miss stats."""

    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        self._store: Dict[str, _CacheEntry] = {}
        self.hits: int = 0
        self.misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, key: str, value: str, ttl: Optional[float] = None) -> None:
        """Store a value with optional TTL override."""
        self._store[key] = _CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached value. Returns None on miss or expiry."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expired:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def invalidate(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()

    def prune_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired_keys = [k for k, v in self._store.items() if v.expired]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size,
        }

    @staticmethod
    def make_key(tool_name: str, args: Dict) -> str:
        """Deterministic cache key for a tool call."""
        import json
        arg_str = json.dumps(args, sort_keys=True)
        return f"{tool_name}:{arg_str}"


# ---------------------------------------------------------------------------
# BatchingAdvisor
# ---------------------------------------------------------------------------

class BatchingAdvisor:
    """Advises whether and how to batch pending tool calls for parallelism."""

    # Minimum number of pending calls before batching is worthwhile
    MIN_BATCH_SIZE = 2

    # Estimated average tool call duration in seconds (for speedup calc)
    _AVG_CALL_DURATION: Dict[str, float] = {
        "read_file": 0.3,
        "mcp_read_file": 0.3,
        "search_files": 0.8,
        "mcp_search_files": 0.8,
        "write_file": 0.4,
        "mcp_write_file": 0.4,
        "patch": 0.5,
        "mcp_patch": 0.5,
        "terminal": 2.0,
        "mcp_terminal": 2.0,
    }

    def __init__(self):
        self._analyzer = DependencyAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_batch(self, pending_calls: List[Dict]) -> bool:
        """Return True if the pending calls would benefit from batching."""
        if len(pending_calls) < self.MIN_BATCH_SIZE:
            return False

        records = self._dicts_to_records(pending_calls)
        groups = self._analyzer.find_parallelizable_groups(records)

        # Batching helps if the first group has more than one call
        if groups and len(groups[0]) >= 2:
            return True

        # Or if total groups < total calls (some parallelism exists)
        total_in_groups = sum(len(g) for g in groups)
        return len(groups) < total_in_groups

    def suggest_batch(self, pending_calls: List[Dict]) -> List[List[Dict]]:
        """Group pending calls into parallel execution batches.

        Returns a list of batches (each batch is a list of call dicts).
        Batches should be executed sequentially; calls within a batch
        can run in parallel.
        """
        if not pending_calls:
            return []

        records = self._dicts_to_records(pending_calls)
        groups = self._analyzer.find_parallelizable_groups(records)

        # Map records back to original dicts by index
        id_to_dict: Dict[str, Dict] = {}
        for rec, orig in zip(records, pending_calls):
            id_to_dict[rec.call_id] = orig

        return [
            [id_to_dict[r.call_id] for r in group if r.call_id in id_to_dict]
            for group in groups
        ]

    def estimate_speedup(self, batch_plan: List[List[Dict]]) -> float:
        """Estimate wall-clock speedup from batching vs sequential.

        Returns a multiplier >= 1.0 (e.g. 2.0 means twice as fast).
        """
        if not batch_plan:
            return 1.0

        sequential_time = 0.0
        parallel_time = 0.0

        for batch in batch_plan:
            durations = [
                self._AVG_CALL_DURATION.get(c.get("tool_name", ""), 1.0)
                for c in batch
            ]
            sequential_time += sum(durations)
            parallel_time += max(durations) if durations else 0.0

        if parallel_time == 0:
            return 1.0

        return round(sequential_time / parallel_time, 2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _dicts_to_records(calls: List[Dict]) -> List[ToolCallRecord]:
        """Convert raw dicts to ToolCallRecord instances."""
        records = []
        for i, c in enumerate(calls):
            records.append(ToolCallRecord(
                tool_name=c.get("tool_name", "unknown"),
                args=c.get("args", {}),
                timestamp=c.get("timestamp", time.time() + i * 0.001),
                result=c.get("result"),
                depends_on=c.get("depends_on", []),
            ))
        return records
