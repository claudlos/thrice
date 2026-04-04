"""Skill index cache with TTL and mtime-based invalidation.

Wraps the existing `build_skills_system_prompt()` machinery in prompt_builder.py
with a proper TTL-based in-memory cache that also checks filesystem mtimes
before serving stale data.  Thread-safe via threading.Lock.

Usage:
    cache = SkillIndexCache(ttl=60)
    prompt = cache.get_skill_index(available_tools=..., available_toolsets=...)

Drop-in replacement for calling `build_skills_system_prompt()` directly.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SkillIndexCache:
    """In-memory cache for the skill index prompt with TTL + mtime invalidation.

    Parameters
    ----------
    ttl : float
        Time-to-live in seconds for cached entries.  Default 60.
    skills_dirs : list[Path] | None
        Skill directories to monitor for changes.  When ``None`` (the default),
        directories are discovered via ``skill_utils.get_all_skills_dirs()`` on
        first access.

    The cache key is derived from (available_tools, available_toolsets) so
    different tool configurations get independent cache entries.
    """

    def __init__(
        self,
        ttl: float = 60.0,
        skills_dirs: Optional[List[Path]] = None,
    ) -> None:
        self._ttl = ttl
        self._explicit_dirs = skills_dirs

        # Cache state
        self._lock = threading.Lock()
        self._cache: Dict[Tuple, str] = {}
        self._timestamps: Dict[Tuple, float] = {}
        self._last_mtime_snapshot: Optional[Dict[str, float]] = None
        self._last_mtime_check: float = 0.0

        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_skill_index(
        self,
        available_tools: Optional[Set[str]] = None,
        available_toolsets: Optional[Set[str]] = None,
    ) -> str:
        """Return the skill index prompt, serving from cache when valid.

        Cache validity requires both:
          1. The entry is younger than *ttl* seconds.
          2. No monitored skill file has a newer mtime than when the cache
             was last populated.
        """
        cache_key = self._make_key(available_tools, available_toolsets)
        now = time.monotonic()

        with self._lock:
            cached = self._cache.get(cache_key)
            ts = self._timestamps.get(cache_key, 0.0)
            age = now - ts

            if cached is not None and age < self._ttl:
                # TTL ok — quick mtime check (at most once per second)
                if not self._mtimes_changed(now):
                    self._hits += 1
                    return cached

            # Cache miss — need a fresh build
            self._misses += 1

        # Build outside the lock to avoid blocking other threads
        result = self._build(available_tools, available_toolsets)
        now = time.monotonic()

        with self._lock:
            self._cache[cache_key] = result
            self._timestamps[cache_key] = now
            self._last_refresh = time.time()
            # Refresh mtime snapshot after build
            self._last_mtime_snapshot = self._collect_mtimes()
            self._last_mtime_check = now

        return result

    def invalidate(self) -> None:
        """Manually bust the entire cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._last_mtime_snapshot = None
            self._last_mtime_check = 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "last_refresh_time": self._last_refresh,
                "cached_entries": len(self._cache),
                "ttl": self._ttl,
            }

    @property
    def ttl(self) -> float:
        return self._ttl

    @ttl.setter
    def ttl(self, value: float) -> None:
        with self._lock:
            self._ttl = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        available_tools: Optional[Set[str]],
        available_toolsets: Optional[Set[str]],
    ) -> Tuple:
        return (
            tuple(sorted(str(t) for t in (available_tools or set()))),
            tuple(sorted(str(ts) for ts in (available_toolsets or set()))),
        )

    def _get_skills_dirs(self) -> List[Path]:
        """Resolve skill directories (lazy, uses explicit list or discovery)."""
        if self._explicit_dirs is not None:
            return self._explicit_dirs
        try:
            from agent.skill_utils import get_all_skills_dirs
            return get_all_skills_dirs()
        except ImportError:
            logger.debug("skill_utils not importable; falling back to ~/.hermes/skills")
            hermes_home = Path.home() / ".hermes"
            skills_dir = hermes_home / "skills"
            return [skills_dir] if skills_dir.exists() else []

    def _collect_mtimes(self) -> Dict[str, float]:
        """Walk skill directories and collect mtime_ns for SKILL.md files."""
        mtimes: Dict[str, float] = {}
        for skills_dir in self._get_skills_dirs():
            if not skills_dir.exists():
                continue
            try:
                for root, dirs, files in os.walk(skills_dir):
                    # Skip hidden/meta dirs
                    dirs[:] = [
                        d for d in dirs
                        if d not in (".git", ".github", ".hub")
                    ]
                    for fname in ("SKILL.md", "DESCRIPTION.md"):
                        if fname in files:
                            p = Path(root) / fname
                            try:
                                mtimes[str(p)] = p.stat().st_mtime_ns
                            except OSError:
                                pass
            except OSError:
                pass
        return mtimes

    def _mtimes_changed(self, now: float) -> bool:
        """Check if any skill file changed since last snapshot.

        Rate-limited to at most once per second to avoid hammering stat().
        """
        if self._last_mtime_snapshot is None:
            return True
        # Rate-limit mtime checks
        if now - self._last_mtime_check < 1.0:
            return False
        self._last_mtime_check = now
        current = self._collect_mtimes()
        return current != self._last_mtime_snapshot

    def _build(
        self,
        available_tools: Optional[Set[str]],
        available_toolsets: Optional[Set[str]],
    ) -> str:
        """Delegate to the real prompt builder."""
        try:
            from agent.prompt_builder import build_skills_system_prompt
            return build_skills_system_prompt(
                available_tools=available_tools,
                available_toolsets=available_toolsets,
            )
        except ImportError:
            logger.warning("Cannot import build_skills_system_prompt; returning empty")
            return ""


# ---------------------------------------------------------------------------
# Module-level singleton for easy integration
# ---------------------------------------------------------------------------

_default_cache: Optional[SkillIndexCache] = None
_default_cache_lock = threading.Lock()


def get_default_cache(ttl: float = 60.0) -> SkillIndexCache:
    """Get or create the module-level singleton cache."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = SkillIndexCache(ttl=ttl)
        return _default_cache


def reset_default_cache() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is not None:
            _default_cache.invalidate()
        _default_cache = None
