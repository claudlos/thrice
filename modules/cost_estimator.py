"""Cost estimator for Hermes Agent (Thrice).

Predicts the token + iteration + wall-clock budget a proposed task is
likely to burn, from historical telemetry captured over previous runs.
Intended to gate the agent loop BEFORE it starts a task that a baseline
already told us would cost 100k tokens.

Two layers:

1. **Per-task history** stored as an append-only JSONL log
   (``~/.hermes/cost_history.jsonl`` by default).  Each record is:
   ``{"task_kind": "...", "tokens": int, "iterations": int,
      "wall_s": float, "ts": epoch, "success": bool}``.

2. **Estimator** with three backends, in priority order:
   - **k-NN on task_kind** (exact or prefix match, weighted by recency).
   - **Feature-based median** (length of description, presence of file
     paths, number of tools referenced).
   - **Static fallback** (conservative defaults).

Public API::

    from cost_estimator import CostEstimator, TaskSpec, CostBudget

    est = CostEstimator()
    est.record(TaskSpec("refactor"), tokens=1200, iterations=3, wall_s=12.0)
    ...
    budget = est.estimate(TaskSpec("refactor"))
    if budget.p95_tokens > 20_000:
        reject("too expensive")
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


DEFAULT_HISTORY_PATH = str(Path.home() / ".hermes" / "cost_history.jsonl")
_RECENCY_HALF_LIFE_S = 14 * 24 * 3600   # 14 days


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    """Inputs to the estimator."""

    kind: str                   # e.g. "refactor", "add_test", "bisect"
    description: str = ""       # natural-language task text
    files_touched: int = 0      # N files expected in blast radius
    tools: Sequence[str] = ()   # tool names likely to be called

    def feature_vector(self) -> Dict[str, float]:
        return {
            "desc_len": float(len(self.description or "")),
            "files": float(self.files_touched),
            "tools": float(len(self.tools)),
            "desc_has_path": 1.0 if any(ch in (self.description or "") for ch in "/\\") else 0.0,
            "desc_has_test": 1.0 if "test" in (self.description or "").lower() else 0.0,
        }


@dataclass
class CostRecord:
    """One past observation."""

    task_kind: str
    tokens: int
    iterations: int
    wall_s: float
    ts: float = field(default_factory=time.time)
    success: bool = True
    description: str = ""

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "CostRecord":
        d = json.loads(line)
        return cls(**d)


@dataclass
class CostBudget:
    """Predicted cost distribution for a task."""

    p50_tokens: int
    p95_tokens: int
    p50_iterations: int
    p95_iterations: int
    p50_wall_s: float
    p95_wall_s: float
    confidence: str             # "high" | "medium" | "low"
    sample_size: int
    method: str                 # "knn_exact" | "knn_prefix" | "features" | "static"

    def is_reasonable(
        self,
        max_tokens: int = 200_000,
        max_iterations: int = 50,
        max_wall_s: float = 1800.0,
    ) -> bool:
        """Return True when the p95 estimate fits the stated ceilings."""
        return (
            self.p95_tokens <= max_tokens
            and self.p95_iterations <= max_iterations
            and self.p95_wall_s <= max_wall_s
        )


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class CostEstimator:
    """Predicts cost based on historical observations."""

    def __init__(
        self,
        history_path: Optional[str] = None,
        static_defaults: Optional[CostBudget] = None,
    ):
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self._records: Optional[List[CostRecord]] = None
        self._static = static_defaults or _DEFAULT_STATIC
        # Serialise writes within a process.  For cross-process safety
        # callers should coordinate out-of-band (e.g. one estimator
        # instance per agent process).
        self._write_lock = threading.Lock()

    # -- History I/O ------------------------------------------------------

    def record(
        self,
        spec: TaskSpec,
        *,
        tokens: int,
        iterations: int,
        wall_s: float,
        success: bool = True,
    ) -> None:
        """Append a new observation to the on-disk history.

        Writes go through ``os.open(O_APPEND | O_CREAT | O_WRONLY)`` so that
        concurrent callers (e.g. two agent processes recording observations
        at once) cannot tear one another's JSONL lines.  On POSIX the
        kernel guarantees atomic append for writes <= PIPE_BUF; on Windows
        the same flag combination gives atomicity for small writes.  We
        keep each record well under that bound by capping the description
        field and emitting a single line per observation.
        """
        rec = CostRecord(
            task_kind=spec.kind,
            tokens=tokens,
            iterations=iterations,
            wall_s=wall_s,
            success=success,
            description=spec.description[:200],
        )
        self._ensure_dir()
        line = rec.to_json_line() + "\n"
        with self._write_lock:
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        # Invalidate the in-memory cache so the next estimate sees the new row.
        self._records = None

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.history_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load(self) -> List[CostRecord]:
        if self._records is not None:
            return self._records
        if not os.path.isfile(self.history_path):
            self._records = []
            return self._records
        out: List[CostRecord] = []
        try:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(CostRecord.from_json_line(line))
                    except Exception:
                        continue
        except OSError:
            pass
        self._records = out
        return out

    # -- Estimation ------------------------------------------------------

    def estimate(self, spec: TaskSpec) -> CostBudget:
        records = self._load()
        exact = [r for r in records if r.task_kind == spec.kind]
        if len(exact) >= 3:
            return self._budget_from(exact, "knn_exact", confidence="high")

        # Prefix match (e.g. "refactor_model" matches "refactor")
        prefix_key = spec.kind.split("_")[0]
        prefix = [r for r in records if r.task_kind.startswith(prefix_key)]
        if len(prefix) >= 5:
            return self._budget_from(prefix, "knn_prefix", confidence="medium")

        # Feature-based fallback: median across all records
        if len(records) >= 5:
            return self._budget_from(records, "features", confidence="low")

        # Static fallback
        return self._static

    # -- Internal --------------------------------------------------------

    def _budget_from(
        self,
        records: Sequence[CostRecord],
        method: str,
        confidence: str,
    ) -> CostBudget:
        weights = [_recency_weight(r.ts) for r in records]
        tokens = _weighted_percentiles([r.tokens for r in records], weights)
        iters = _weighted_percentiles([r.iterations for r in records], weights)
        walls = _weighted_percentiles([r.wall_s for r in records], weights)
        return CostBudget(
            p50_tokens=int(tokens[0]),
            p95_tokens=int(tokens[1]),
            p50_iterations=int(iters[0]),
            p95_iterations=int(iters[1]),
            p50_wall_s=float(walls[0]),
            p95_wall_s=float(walls[1]),
            confidence=confidence,
            sample_size=len(records),
            method=method,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recency_weight(ts: float, now: Optional[float] = None) -> float:
    """Exponential-decay weight so old records count less than fresh ones."""
    now = now if now is not None else time.time()
    age = max(0.0, now - ts)
    return 0.5 ** (age / _RECENCY_HALF_LIFE_S)


def _weighted_percentiles(values: Sequence[float], weights: Sequence[float]) -> tuple:
    """Return ``(p50, p95)`` using weighted samples.  Falls back to plain
    percentiles if weights sum to ~0."""
    if not values:
        return (0.0, 0.0)
    total = sum(weights)
    if total < 1e-9:
        sorted_v = sorted(values)
        return (statistics.median(sorted_v), sorted_v[int(len(sorted_v) * 0.95)])
    pairs = sorted(zip(values, weights, strict=True))
    cum = 0.0
    p50: Optional[float] = None
    p95: Optional[float] = None
    for v, w in pairs:
        cum += w
        pct = cum / total
        if p50 is None and pct >= 0.5:
            p50 = float(v)
        if p95 is None and pct >= 0.95:
            p95 = float(v)
            break
    if p50 is None:
        p50 = float(pairs[-1][0])
    if p95 is None:
        p95 = float(pairs[-1][0])
    return (p50, p95)


_DEFAULT_STATIC = CostBudget(
    p50_tokens=2_000,    p95_tokens=20_000,
    p50_iterations=2,    p95_iterations=10,
    p50_wall_s=20.0,     p95_wall_s=180.0,
    confidence="low",    sample_size=0,
    method="static",
)


__all__ = [
    "CostBudget",
    "CostEstimator",
    "CostRecord",
    "TaskSpec",
]
