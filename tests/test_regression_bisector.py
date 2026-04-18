"""Tests for ``regression_bisector``.

Each test builds a throw-away git repo with a known regression and
drives the bisector against it, so the suite works on any machine with
``git`` on PATH.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from regression_bisector import (  # noqa: E402
    BisectConfig,
    Bisector,
    BisectResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_env(tmp_parent: str) -> dict:
    """Env that isolates tests from the user's git config/outer repos."""
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        # Stop git from walking up into the developer's own repo.
        "GIT_CEILING_DIRECTORIES": tmp_parent,
        # Ignore per-user config (e.g. signing.gpg.sign=true).
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }


def _git(*args: str, cwd: str, env: dict) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed (rc={r.returncode})\n"
            f"stdout: {r.stdout}\nstderr: {r.stderr}"
        )
    return r.stdout.strip()


@pytest.fixture
def tiny_repo(tmp_path):
    """Seven commits: c0..c6.  c3 introduces the regression."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _git_env(str(tmp_path))
    _git("init", "-q", "-b", "main", cwd=str(repo), env=env)
    _git("config", "commit.gpgsign", "false", cwd=str(repo), env=env)
    _git("config", "user.name", "t", cwd=str(repo), env=env)
    _git("config", "user.email", "t@t", cwd=str(repo), env=env)
    bad_from = 3
    commits: list[str] = []
    for i in range(7):
        state = "broken" if i >= bad_from else "ok"
        # Always include ``i`` so each commit has a distinct diff
        # (otherwise "good → good" commits would be "nothing to commit").
        (repo / "value.txt").write_text(f"commit={i} state={state}\n")
        _git("add", "value.txt", cwd=str(repo), env=env)
        _git("commit", "-m", f"c{i}", cwd=str(repo), env=env)
        commits.append(_git("rev-parse", "HEAD", cwd=str(repo), env=env))

    # Test script: exit 0 if value.txt's state marker is "ok", else exit 1.
    test_script = repo / "test.py"
    test_script.write_text(
        "from pathlib import Path; import sys;"
        "sys.exit(0 if 'state=ok' in Path('value.txt').read_text() else 1)\n"
    )
    return repo, commits, bad_from


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_finds_the_first_bad_commit(self, tiny_repo):
        repo, commits, bad_idx = tiny_repo
        cfg = BisectConfig(
            repo_dir=str(repo),
            good=commits[0],
            bad=commits[-1],
            test_cmd=[sys.executable, "test.py"],
        )
        result = Bisector(cfg).run()
        assert isinstance(result, BisectResult)
        assert result.ok, result.error
        assert result.first_bad_commit == commits[bad_idx]
        assert result.state == "completed"
        # At least log(7) ~ 3 steps.
        assert 1 <= len(result.steps) <= 7
        # Every recorded step has a classification.
        for step in result.steps:
            assert step.outcome in ("good", "bad", "skip")

    def test_returns_deterministic_sha_format(self, tiny_repo):
        repo, commits, bad_idx = tiny_repo
        cfg = BisectConfig(
            repo_dir=str(repo),
            good=commits[0],
            bad=commits[-1],
            test_cmd=[sys.executable, "test.py"],
        )
        result = Bisector(cfg).run()
        # git bisect reports the full sha.
        assert result.first_bad_commit and len(result.first_bad_commit) == 40


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_non_git_dir_aborts_cleanly(self, tmp_path, monkeypatch):
        # Prevent git from walking up to the developer's outer repo.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        nogit = tmp_path / "nogit"
        nogit.mkdir()
        cfg = BisectConfig(
            repo_dir=str(nogit),
            good="HEAD",
            bad="HEAD",
            test_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
        )
        result = Bisector(cfg).run()
        assert not result.ok
        assert result.state == "aborted"

    def test_bad_commit_aborts_cleanly(self, tiny_repo):
        repo, commits, _ = tiny_repo
        cfg = BisectConfig(
            repo_dir=str(repo),
            good=commits[0],
            bad="nonexistent-ref",
            test_cmd=[sys.executable, "test.py"],
        )
        result = Bisector(cfg).run()
        assert not result.ok
        assert result.state == "aborted"

    def test_timeout_per_step(self, tiny_repo, monkeypatch):
        repo, commits, _ = tiny_repo
        # Hang script: sleeps longer than the step timeout.
        (repo / "slow.py").write_text("import time; time.sleep(5)\n")
        cfg = BisectConfig(
            repo_dir=str(repo),
            good=commits[0],
            bad=commits[-1],
            test_cmd=[sys.executable, "slow.py"],
            step_timeout=0.5,
            skip_rc=125,
        )
        # Sleeping script will be killed as "bad" (exit != 0).  We don't
        # assert a specific commit - just that the loop terminates.
        result = Bisector(cfg).run()
        # Either converged quickly, or aborted without hanging the test suite.
        assert result.state in ("completed", "aborted")


# ---------------------------------------------------------------------------
# Unit: output parser
# ---------------------------------------------------------------------------

class TestOutputParser:
    def test_extracts_sha(self):
        out = (
            "Bisecting: 0 revisions left to test after this (roughly 0 steps)\n"
            "abc123def456abc123def456abc123def4561234 is the first bad commit\n"
            "commit abc123def456abc123def456abc123def4561234\n"
        )
        assert Bisector._extract_first_bad_commit(out) == \
            "abc123def456abc123def456abc123def4561234"

    def test_returns_none_when_no_match(self):
        assert Bisector._extract_first_bad_commit("no match here") is None
