"""Tests for SkillIndexCache — TTL, mtime invalidation, thread safety, stats."""

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make new-files importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "new-files"))

from skill_index_cache import SkillIndexCache, get_default_cache, reset_default_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_skills(tmp_path):
    """Create a minimal skills directory with two skills."""
    skills_dir = tmp_path / "skills"

    # Skill 1
    s1 = skills_dir / "devops" / "deploy-aws"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(
        "---\nname: deploy-aws\ndescription: Deploy to AWS\n---\nDeploy steps...\n"
    )

    # Skill 2
    s2 = skills_dir / "devops" / "deploy-gcp"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(
        "---\nname: deploy-gcp\ndescription: Deploy to GCP\n---\nDeploy steps...\n"
    )

    # Category description
    desc = skills_dir / "devops"
    (desc / "DESCRIPTION.md").write_text(
        "---\ndescription: DevOps deployment skills\n---\n"
    )

    return skills_dir


@pytest.fixture
def cache(tmp_skills):
    """A SkillIndexCache pointing at the temp skills dir with short TTL."""
    return SkillIndexCache(ttl=2.0, skills_dirs=[tmp_skills])


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestSkillIndexCache:

    def test_fresh_build_returns_string(self, cache):
        """First call is a cache miss and delegates to _build."""
        with patch.object(cache, "_build", return_value="## Skills prompt") as mock_build:
            result = cache.get_skill_index()
            assert result == "## Skills prompt"
            mock_build.assert_called_once()

    def test_second_call_hits_cache(self, cache):
        """Second call within TTL should be a cache hit."""
        with patch.object(cache, "_build", return_value="cached prompt") as mock_build:
            r1 = cache.get_skill_index()
            r2 = cache.get_skill_index()
            assert r1 == r2 == "cached prompt"
            assert mock_build.call_count == 1

    def test_ttl_expiry_triggers_rebuild(self, cache):
        """After TTL expires, a rebuild should happen."""
        cache._ttl = 0.1  # 100ms TTL

        with patch.object(cache, "_build", return_value="v1") as mock_build:
            r1 = cache.get_skill_index()
            assert r1 == "v1"
            assert mock_build.call_count == 1

        time.sleep(0.15)

        with patch.object(cache, "_build", return_value="v2") as mock_build:
            r2 = cache.get_skill_index()
            assert r2 == "v2"
            assert mock_build.call_count == 1

    def test_different_tools_different_entries(self, cache):
        """Different tool sets should get independent cache entries."""
        call_count = 0

        def mock_build(tools, toolsets):
            nonlocal call_count
            call_count += 1
            return f"prompt_{call_count}"

        with patch.object(cache, "_build", side_effect=mock_build):
            r1 = cache.get_skill_index(available_tools={"tool_a"})
            r2 = cache.get_skill_index(available_tools={"tool_b"})
            assert r1 != r2
            assert call_count == 2

    def test_invalidate_clears_cache(self, cache):
        """invalidate() should force a rebuild on next call."""
        with patch.object(cache, "_build", return_value="before") as mock_build:
            cache.get_skill_index()
            assert mock_build.call_count == 1

        cache.invalidate()

        with patch.object(cache, "_build", return_value="after") as mock_build:
            result = cache.get_skill_index()
            assert result == "after"
            assert mock_build.call_count == 1


# ---------------------------------------------------------------------------
# Mtime-based invalidation
# ---------------------------------------------------------------------------

class TestMtimeInvalidation:

    def test_mtime_change_busts_cache(self, cache, tmp_skills):
        """Changing a SKILL.md file mtime should invalidate the cache."""
        with patch.object(cache, "_build", return_value="original"):
            cache.get_skill_index()

        # Modify a skill file
        skill_file = tmp_skills / "devops" / "deploy-aws" / "SKILL.md"
        skill_file.write_text("---\nname: deploy-aws\ndescription: Updated\n---\n")

        # Force mtime check cooldown to pass
        cache._last_mtime_check = 0.0

        with patch.object(cache, "_build", return_value="refreshed") as mock_build:
            result = cache.get_skill_index()
            assert result == "refreshed"
            mock_build.assert_called_once()

    def test_new_skill_file_busts_cache(self, cache, tmp_skills):
        """Adding a new SKILL.md should invalidate the cache."""
        with patch.object(cache, "_build", return_value="original"):
            cache.get_skill_index()

        # Add a new skill
        new_skill = tmp_skills / "forged" / "new-skill"
        new_skill.mkdir(parents=True)
        (new_skill / "SKILL.md").write_text("---\nname: new-skill\n---\n")

        # Force mtime check cooldown to pass
        cache._last_mtime_check = 0.0

        with patch.object(cache, "_build", return_value="with-new-skill") as mock_build:
            result = cache.get_skill_index()
            assert result == "with-new-skill"
            mock_build.assert_called_once()

    def test_deleted_skill_file_busts_cache(self, cache, tmp_skills):
        """Removing a SKILL.md should invalidate the cache."""
        with patch.object(cache, "_build", return_value="original"):
            cache.get_skill_index()

        # Delete a skill
        skill_file = tmp_skills / "devops" / "deploy-gcp" / "SKILL.md"
        skill_file.unlink()

        cache._last_mtime_check = 0.0

        with patch.object(cache, "_build", return_value="after-delete") as mock_build:
            result = cache.get_skill_index()
            assert result == "after-delete"
            mock_build.assert_called_once()

    def test_mtime_check_rate_limited(self, cache, tmp_skills):
        """Mtime checks should be rate-limited (at most once per second)."""
        with patch.object(cache, "_build", return_value="prompt"):
            cache.get_skill_index()

        # Immediately after, _collect_mtimes should NOT be called
        with patch.object(cache, "_collect_mtimes") as mock_collect:
            with patch.object(cache, "_build", return_value="prompt"):
                cache.get_skill_index()
            mock_collect.assert_not_called()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_initial_stats(self, cache):
        """Fresh cache should have zero hits and misses."""
        s = cache.stats()
        assert s["cache_hits"] == 0
        assert s["cache_misses"] == 0
        assert s["cached_entries"] == 0
        assert s["ttl"] == 2.0

    def test_stats_after_operations(self, cache):
        """Stats should reflect actual cache behavior."""
        with patch.object(cache, "_build", return_value="prompt"):
            cache.get_skill_index()  # miss
            cache.get_skill_index()  # hit
            cache.get_skill_index()  # hit

        s = cache.stats()
        assert s["cache_misses"] == 1
        assert s["cache_hits"] == 2
        assert s["cached_entries"] == 1
        assert s["last_refresh_time"] > 0

    def test_stats_after_invalidate(self, cache):
        """Invalidation should clear entries but preserve cumulative stats."""
        with patch.object(cache, "_build", return_value="prompt"):
            cache.get_skill_index()
            cache.get_skill_index()

        cache.invalidate()
        s = cache.stats()
        assert s["cached_entries"] == 0
        # Cumulative stats preserved
        assert s["cache_hits"] == 1
        assert s["cache_misses"] == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_access(self, cache):
        """Multiple threads should safely access the cache."""
        build_count = 0
        build_lock = threading.Lock()

        def slow_build(tools, toolsets):
            nonlocal build_count
            time.sleep(0.01)  # Simulate work
            with build_lock:
                build_count += 1
            return "concurrent_prompt"

        results = []
        errors = []

        def worker():
            try:
                with patch.object(cache, "_build", side_effect=slow_build):
                    r = cache.get_skill_index()
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert all(r == "concurrent_prompt" for r in results)

    def test_concurrent_invalidate(self, cache):
        """invalidate() from one thread shouldn't crash others."""
        errors = []

        def reader():
            try:
                for _ in range(20):
                    with patch.object(cache, "_build", return_value="p"):
                        cache.get_skill_index()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def invalidator():
            try:
                for _ in range(10):
                    cache.invalidate()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=invalidator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# TTL configuration
# ---------------------------------------------------------------------------

class TestTTLConfig:

    def test_ttl_property(self, cache):
        assert cache.ttl == 2.0
        cache.ttl = 30.0
        assert cache.ttl == 30.0

    def test_zero_ttl_always_rebuilds(self, tmp_skills):
        """TTL of 0 should always miss."""
        cache = SkillIndexCache(ttl=0, skills_dirs=[tmp_skills])
        call_count = 0

        def mock_build(tools, toolsets):
            nonlocal call_count
            call_count += 1
            return f"v{call_count}"

        with patch.object(cache, "_build", side_effect=mock_build):
            r1 = cache.get_skill_index()
            r2 = cache.get_skill_index()
            # Both should trigger builds (TTL=0 means always expired)
            assert call_count == 2


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_default_cache_returns_same_instance(self):
        reset_default_cache()
        c1 = get_default_cache(ttl=30)
        c2 = get_default_cache(ttl=30)
        assert c1 is c2
        reset_default_cache()

    def test_reset_clears_singleton(self):
        reset_default_cache()
        c1 = get_default_cache()
        reset_default_cache()
        c2 = get_default_cache()
        assert c1 is not c2
        reset_default_cache()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_nonexistent_skills_dir(self, tmp_path):
        """Cache should handle missing skills directories gracefully."""
        cache = SkillIndexCache(
            ttl=60,
            skills_dirs=[tmp_path / "nonexistent"],
        )
        with patch.object(cache, "_build", return_value=""):
            result = cache.get_skill_index()
            assert result == ""

    def test_empty_skills_dir(self, tmp_path):
        """Cache should handle an empty skills directory."""
        empty_dir = tmp_path / "skills"
        empty_dir.mkdir()
        cache = SkillIndexCache(ttl=60, skills_dirs=[empty_dir])
        with patch.object(cache, "_build", return_value=""):
            result = cache.get_skill_index()
            assert result == ""

    def test_collect_mtimes_with_real_files(self, cache, tmp_skills):
        """_collect_mtimes should find actual SKILL.md files."""
        mtimes = cache._collect_mtimes()
        assert len(mtimes) >= 2  # At least our 2 SKILL.md + 1 DESCRIPTION.md
        for path_str in mtimes:
            assert "SKILL.md" in path_str or "DESCRIPTION.md" in path_str
