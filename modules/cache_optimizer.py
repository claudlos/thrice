"""KV-cache hit-rate tracker + prefix-stability enforcer for Hermes Agent (Thrice).

Grounded in:
    - Manus engineering blog (2024), "Context Engineering for AI Agents":
      *"KV-cache hit rate is the single most important metric for a
      production-stage AI agent."*
    - "Don't Break the Cache: An Evaluation of Prompt Caching for
      Long-Horizon Agentic Tasks" (arXiv:2601.06007, 2026): measured
      45-80 % cost reduction and 13-31 % TTFT improvement from well-placed
      caching; dynamic prefixes / tool-definition churn identified as
      primary cache-kill patterns.

Two complementary primitives:

1. ``PrefixGuard`` — detects prefix-invalidating mutations *before* a
   request goes out, so the caller can repair them instead of silently
   paying for a cold call.  Watches for:

   * timestamps, UUIDs, absolute-time fields, session IDs injected into
     the system prompt or system-message prefix;
   * tool-definition churn (add / remove / reorder) vs. the last turn;
   * content that moved earlier in the prefix (cache misses if any byte
     before the boundary changes).

2. ``CacheTracker`` — records per-turn cache statistics (tokens in,
   cached, hit rate, estimated savings) and surfaces regressions so the
   agent loop can decide to recompact or bail on degenerate prefix
   mutation before it burns the session budget.

Both primitives are provider-neutral; wire them up to whichever SDK
exposes cache-usage telemetry (OpenAI ``prompt_cache_hit_tokens``,
Anthropic ``cache_read_input_tokens``, Google equivalent).

Typical wiring::

    from cache_optimizer import PrefixGuard, CacheTracker

    guard   = PrefixGuard()
    tracker = CacheTracker()

    # Before the request
    verdict = guard.check(system_prompt, tools, prefix_messages)
    if verdict.has_breakage:
        log.warning("cache-kill: %s", verdict.summary())

    # After the response
    tracker.record(
        turn=turn_idx,
        input_tokens=usage["input_tokens"],
        cached_tokens=usage.get("cache_read_input_tokens", 0),
        provider="anthropic",
    )
    if tracker.needs_attention():
        print(tracker.format_summary())
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefix-stability guard
# ---------------------------------------------------------------------------

# Patterns that almost always sink a prefix cache.  If these appear *before*
# the last cache boundary the next turn has no chance of a hit.
_DYNAMIC_PATTERNS = (
    # ISO-8601 timestamps
    (re.compile(r"\b20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "timestamp"),
    # RFC-822ish dates
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM|UTC)?"), "time"),
    # UUIDs
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
     "uuid"),
    # Unix epoch seconds (10 digits, ~1970-2286)
    (re.compile(r"\b1[6-9]\d{8}\b"), "epoch_seconds"),
    # Unix epoch millis (13 digits)
    (re.compile(r"\b1[6-9]\d{11}\b"), "epoch_millis"),
    # Session / request / correlation ids (typed prefixes)
    (re.compile(r"\b(?:session|request|correlation|trace|span)[-_]?id[\s:=]*"
                r"['\"]?[A-Za-z0-9_-]{6,}", re.I),
     "session_id"),
)


@dataclass(frozen=True)
class PrefixBreakage:
    """One reason the prefix has likely lost the KV cache."""

    kind: str         # e.g. "timestamp", "uuid", "tools_reordered"
    detail: str       # human-readable description
    location: str     # "system_prompt" | "tools" | "messages[0]" | ...

    def format_short(self) -> str:
        return f"[{self.location}] {self.kind}: {self.detail}"


@dataclass
class PrefixVerdict:
    """Result of a single prefix stability check."""

    ok: bool
    breakages: List[PrefixBreakage] = field(default_factory=list)

    @property
    def has_breakage(self) -> bool:
        return bool(self.breakages)

    def summary(self) -> str:
        if self.ok:
            return "prefix: stable"
        return "prefix breakages: " + "; ".join(b.format_short() for b in self.breakages[:5])


class PrefixGuard:
    """Detects cache-invalidating mutations in the stable prefix.

    A guard instance is stateful so it can compare the current prefix
    against the previous one (for tool-set churn detection).  Thread-safe
    only if external callers don't share the same instance across threads.
    """

    def __init__(self):
        # Set hash detects add/remove (order-insensitive); order hash
        # detects reorder-only churn (sequence-sensitive).  Keeping them
        # separate lets us emit precise diagnostics.
        self._last_tools_set_hash: Optional[str] = None
        self._last_tools_order_hash: Optional[str] = None
        self._last_tool_names: Tuple[str, ...] = ()
        self._last_system_hash: Optional[str] = None

    # -- Public API -------------------------------------------------------

    def check(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        prefix_messages: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> PrefixVerdict:
        """Check the prefix for known cache-kill patterns."""
        breakages: List[PrefixBreakage] = []

        if system_prompt is not None:
            breakages.extend(self._scan_text(system_prompt, "system_prompt"))

        if prefix_messages:
            for i, msg in enumerate(prefix_messages):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                breakages.extend(self._scan_text(content, f"messages[{i}]"))

        if tools is not None:
            breakages.extend(self._check_tool_stability(tools))

        if system_prompt is not None:
            cur = _hash(system_prompt)
            if self._last_system_hash and cur != self._last_system_hash:
                breakages.append(PrefixBreakage(
                    kind="system_prompt_changed",
                    detail="system prompt bytes differ from previous turn",
                    location="system_prompt",
                ))
            self._last_system_hash = cur

        return PrefixVerdict(ok=not breakages, breakages=breakages)

    # -- Internals --------------------------------------------------------

    def _scan_text(self, text: str, location: str) -> List[PrefixBreakage]:
        out: List[PrefixBreakage] = []
        for pat, kind in _DYNAMIC_PATTERNS:
            m = pat.search(text)
            if m:
                out.append(PrefixBreakage(
                    kind=kind,
                    detail=f"dynamic value found: {m.group(0)!r}",
                    location=location,
                ))
        return out

    def _check_tool_stability(self, tools: Sequence[Dict[str, Any]]) -> List[PrefixBreakage]:
        names = tuple(t.get("name", "") for t in tools)
        # Two hashes so we can distinguish "set churn" from "order churn".
        # Both are cache-invalidating but the diagnostic is different —
        # reorders are easy to fix, adds/removes often aren't.
        set_hash   = _hash("\x00".join(sorted(names)))
        order_hash = _hash("\x00".join(names))
        out: List[PrefixBreakage] = []
        if self._last_tools_set_hash is not None:
            prev_set = set(self._last_tool_names)
            curr_set = set(names)
            if curr_set != prev_set:
                added   = curr_set - prev_set
                removed = prev_set - curr_set
                detail_parts = []
                if added:
                    detail_parts.append(f"added={sorted(added)}")
                if removed:
                    detail_parts.append(f"removed={sorted(removed)}")
                out.append(PrefixBreakage(
                    kind="tools_changed",
                    detail=", ".join(detail_parts),
                    location="tools",
                ))
            elif order_hash != self._last_tools_order_hash:
                # Same set, different order — every byte shifts, cache dies.
                out.append(PrefixBreakage(
                    kind="tools_reordered",
                    detail=(
                        f"order changed: {list(self._last_tool_names)} -> "
                        f"{list(names)}"
                    ),
                    location="tools",
                ))
        self._last_tools_set_hash   = set_hash
        self._last_tools_order_hash = order_hash
        self._last_tool_names       = names
        return out


# ---------------------------------------------------------------------------
# Cache-usage tracker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheTurn:
    """Per-turn cache statistics."""

    turn: int
    input_tokens: int
    cached_tokens: int
    provider: str
    ts: float

    @property
    def hit_rate(self) -> float:
        return (self.cached_tokens / self.input_tokens) if self.input_tokens else 0.0


# Rough dollar / latency coefficients for "Don't Break the Cache" math.
# Tuned to report savings with the paper's reported 45-80 % band; callers
# who need precision should pass their own ``pricing=`` kwargs.
_DEFAULT_PRICING = {
    # $ per million input tokens, (full, cached)
    "anthropic": (3.0, 0.30),
    "openai":    (2.5, 0.25),
    "google":    (1.5, 0.15),
    "generic":   (2.0, 0.20),
}


@dataclass
class CacheStats:
    """Aggregate cache statistics over a session / window."""

    turns: int
    total_input: int
    total_cached: int
    mean_hit_rate: float
    p50_hit_rate: float
    p95_hit_rate: float
    cost_nominal: float          # $ assuming no caching
    cost_actual: float           # $ with the observed cache usage
    savings: float               # cost_nominal - cost_actual
    savings_pct: float           # 0..1

    def format_short(self) -> str:
        return (
            f"cache: {self.turns} turns, "
            f"mean hit {self.mean_hit_rate * 100:.1f}%, "
            f"p50 {self.p50_hit_rate * 100:.1f}%, "
            f"savings ${self.savings:.3f} ({self.savings_pct * 100:.1f}%)"
        )


class CacheTracker:
    """Records per-turn cache telemetry and surfaces regressions."""

    def __init__(
        self,
        *,
        provider_pricing: Optional[Dict[str, Tuple[float, float]]] = None,
        regression_threshold: float = 0.20,
        window: int = 10,
    ):
        self._turns: List[CacheTurn] = []
        self._pricing = provider_pricing or _DEFAULT_PRICING
        self._regression_threshold = regression_threshold
        self._window = window

    # -- Ingest -----------------------------------------------------------

    def record(
        self,
        *,
        turn: int,
        input_tokens: int,
        cached_tokens: int,
        provider: str = "generic",
        ts: Optional[float] = None,
    ) -> CacheTurn:
        """Record one turn's cache usage; returns the stored record."""
        rec = CacheTurn(
            turn=turn,
            input_tokens=max(0, int(input_tokens)),
            cached_tokens=max(0, int(cached_tokens)),
            provider=provider,
            ts=ts if ts is not None else time.time(),
        )
        if rec.cached_tokens > rec.input_tokens:
            # Some providers report cache_read > input; clamp.
            rec = CacheTurn(
                turn=rec.turn, input_tokens=rec.input_tokens,
                cached_tokens=rec.input_tokens,
                provider=rec.provider, ts=rec.ts,
            )
        self._turns.append(rec)
        return rec

    # -- Query ------------------------------------------------------------

    def stats(self) -> CacheStats:
        if not self._turns:
            return CacheStats(
                turns=0, total_input=0, total_cached=0,
                mean_hit_rate=0.0, p50_hit_rate=0.0, p95_hit_rate=0.0,
                cost_nominal=0.0, cost_actual=0.0,
                savings=0.0, savings_pct=0.0,
            )
        hit_rates = [t.hit_rate for t in self._turns]
        total_in = sum(t.input_tokens for t in self._turns)
        total_cached = sum(t.cached_tokens for t in self._turns)
        cost_nominal = 0.0
        cost_actual = 0.0
        for t in self._turns:
            full, cached = self._pricing.get(t.provider, self._pricing["generic"])
            # Prices are $/million tokens.
            cost_nominal += (t.input_tokens / 1_000_000) * full
            cost_actual += (
                ((t.input_tokens - t.cached_tokens) / 1_000_000) * full
                + (t.cached_tokens / 1_000_000) * cached
            )
        savings = cost_nominal - cost_actual
        return CacheStats(
            turns=len(self._turns),
            total_input=total_in,
            total_cached=total_cached,
            mean_hit_rate=statistics.fmean(hit_rates),
            p50_hit_rate=statistics.median(hit_rates),
            p95_hit_rate=_percentile(hit_rates, 0.95),
            cost_nominal=cost_nominal,
            cost_actual=cost_actual,
            savings=savings,
            savings_pct=(savings / cost_nominal) if cost_nominal > 0 else 0.0,
        )

    def needs_attention(self) -> bool:
        """True when the last ``window`` turns show a regression vs. the
        rolling baseline.  Cheap heuristic: recent-mean hit rate fell by
        more than ``regression_threshold`` compared to the earlier
        session mean."""
        if len(self._turns) < 2 * self._window:
            return False
        recent = self._turns[-self._window:]
        earlier = self._turns[:-self._window]
        recent_mean = statistics.fmean(t.hit_rate for t in recent)
        earlier_mean = statistics.fmean(t.hit_rate for t in earlier)
        return (earlier_mean - recent_mean) >= self._regression_threshold

    def format_summary(self) -> str:
        return self.stats().format_short()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


__all__ = [
    "CacheStats",
    "CacheTracker",
    "CacheTurn",
    "PrefixBreakage",
    "PrefixGuard",
    "PrefixVerdict",
]
