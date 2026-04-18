"""Tests for auto_commit.py — Auto-commit + /undo system."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))
from auto_commit import (
    DEFAULT_PREFIX,
    AutoCommitManager,
    GitError,
    UndoManager,
    _run_git,
    is_auto_commit_enabled,
)


def _init_git_repo(path: str):
    """Initialize a git repo with an initial commit."""
    subprocess.run(["git", "init", path], capture_output=True, check=True)
    subprocess.run(["git", "-C", path, "config", "user.email", "test@test.com"],
                    capture_output=True, check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Test"],
                    capture_output=True, check=True)
    # Initial commit
    readme = os.path.join(path, "README.md")
    with open(readme, "w") as f:
        f.write("# Test\n")
    subprocess.run(["git", "-C", path, "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", path, "commit", "-m", "initial commit"],
                    capture_output=True, check=True)
    # Create and stay on a non-main branch for safety tests
    subprocess.run(["git", "-C", path, "checkout", "-b", "dev"],
                    capture_output=True, check=True)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    _init_git_repo(repo)
    return repo


class TestAutoCommitManager:
    def test_enable_on_valid_repo(self, git_repo):
        mgr = AutoCommitManager()
        assert mgr.enable(git_repo) is True
        assert mgr.enabled is True

    def test_enable_on_non_git_dir(self, tmp_path):
        mgr = AutoCommitManager()
        assert mgr.enable(str(tmp_path)) is False
        assert mgr.enabled is False

    def test_disable(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)
        mgr.disable()
        assert mgr.enabled is False

    def test_on_file_edit_creates_commit(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)

        # Create a new file
        test_file = os.path.join(git_repo, "hello.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")

        record = mgr.on_file_edit(test_file, "Added hello.py")
        assert record is not None
        assert record.sha
        assert "hermes:" in record.message
        assert len(mgr.session_commits) == 1

    def test_on_file_edit_when_disabled(self, git_repo):
        mgr = AutoCommitManager()
        # Don't enable
        test_file = os.path.join(git_repo, "hello.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")

        record = mgr.on_file_edit(test_file, "Added hello.py")
        assert record is None

    def test_multiple_commits(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)

        for i in range(3):
            path = os.path.join(git_repo, f"file{i}.py")
            with open(path, "w") as f:
                f.write(f"# file {i}\n")
            mgr.on_file_edit(path, f"Added file{i}.py")

        assert len(mgr.session_commits) == 3

    def test_main_branch_creates_feature_branch(self, tmp_path):
        """When on main, auto-commit should create a feature branch."""
        repo = str(tmp_path / "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-b", "main", repo], capture_output=True, check=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "t@t.com"],
                        capture_output=True, check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "T"],
                        capture_output=True, check=True)
        readme = os.path.join(repo, "README.md")
        with open(readme, "w") as f:
            f.write("# Test\n")
        subprocess.run(["git", "-C", repo, "add", "."], capture_output=True, check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "init"],
                        capture_output=True, check=True)

        mgr = AutoCommitManager()
        assert mgr.enable(repo) is True

        # Should be on a hermes/ branch now
        result = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True)
        assert result.stdout.strip().startswith("hermes/")

    def test_detached_head_prevents_enable(self, git_repo):
        """Should not enable on detached HEAD."""
        # Detach HEAD
        subprocess.run(["git", "-C", git_repo, "checkout", "--detach"],
                        capture_output=True, check=True)
        mgr = AutoCommitManager()
        assert mgr.enable(git_repo) is False

    def test_no_changes_returns_none(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)
        # Edit existing tracked file to same content (no actual change)
        record = mgr.on_file_edit(os.path.join(git_repo, "README.md"), "no-op")
        assert record is None

    def test_custom_prefix(self, git_repo):
        mgr = AutoCommitManager(prefix="ai-edit:")
        mgr.enable(git_repo)

        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "w") as f:
            f.write("test\n")

        record = mgr.on_file_edit(test_file, "test file")
        assert record is not None
        assert record.message.startswith("ai-edit:")


class TestUndoManager:
    def test_undo_last_ai_commit(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)

        test_file = os.path.join(git_repo, "hello.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")
        mgr.on_file_edit(test_file, "Added hello.py")

        undo = UndoManager(git_repo, session_commits=mgr.session_commits)
        sha = undo.undo()
        assert sha is not None
        # File should be removed after undo
        assert not os.path.exists(test_file)

    def test_undo_non_ai_commit(self, git_repo):
        """undo() should return None if HEAD is not an AI commit."""
        undo = UndoManager(git_repo)
        assert undo.undo() is None

    def test_undo_all(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)

        for i in range(3):
            path = os.path.join(git_repo, f"file{i}.py")
            with open(path, "w") as f:
                f.write(f"# {i}\n")
            mgr.on_file_edit(path, f"file {i}")

        undo = UndoManager(git_repo, session_commits=mgr.session_commits)
        reverted = undo.undo_all()
        # At least some should be reverted
        assert len(reverted) >= 1

    def test_get_ai_commits(self, git_repo):
        mgr = AutoCommitManager()
        mgr.enable(git_repo)

        for i in range(2):
            path = os.path.join(git_repo, f"f{i}.txt")
            with open(path, "w") as f:
                f.write(f"{i}\n")
            mgr.on_file_edit(path, f"edit {i}")

        undo = UndoManager(git_repo)
        commits = undo.get_ai_commits()
        assert len(commits) == 2
        for c in commits:
            assert c.message.startswith(DEFAULT_PREFIX)

    def test_diff_no_changes(self, git_repo):
        undo = UndoManager(git_repo)
        result = undo.diff()
        assert "No uncommitted changes" in result

    def test_diff_with_changes(self, git_repo):
        # Modify a tracked file without committing
        readme = os.path.join(git_repo, "README.md")
        with open(readme, "a") as f:
            f.write("more content\n")

        undo = UndoManager(git_repo)
        result = undo.diff()
        assert "more content" in result


class TestConfiguration:
    def test_env_var_true(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_COMMIT", "true")
        assert is_auto_commit_enabled() is True

    def test_env_var_false(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_COMMIT", "false")
        assert is_auto_commit_enabled() is False

    def test_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_AUTO_COMMIT", raising=False)
        # Without config file, defaults to False
        assert is_auto_commit_enabled() is False


class TestGitError:
    def test_run_git_failure(self, tmp_path, monkeypatch):
        # ``tmp_path`` may live under a directory that is itself a git repo
        # (e.g. a developer home dir), so ``git status`` there succeeds by
        # walking up to the outer repo.  Cap the search with
        # ``GIT_CEILING_DIRECTORIES`` so git really cannot find one.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        with pytest.raises(GitError):
            _run_git(["status"], str(tmp_path))


# ============================================================================
# Integration: secret_scanner + conventional_commit wiring
# ============================================================================

class TestSecretScanIntegration:
    """on_file_edit must block when the staged diff contains a secret."""

    def test_secret_blocks_commit(self, git_repo):
        mgr = AutoCommitManager(scan_for_secrets=True, conventional_messages=False)
        assert mgr.enable(git_repo)
        # Plant a high-severity secret.
        secret_path = os.path.join(git_repo, "settings.py")
        with open(secret_path, "w") as f:
            f.write("AWS_KEY = 'AKIAABCDEFGHIJKLMNOP'\n")

        record = mgr.on_file_edit(secret_path, description="add settings")
        assert record is None, "commit should have been blocked"
        assert mgr.last_blocked_findings, "expected at least one finding"
        # File should be unstaged after block.
        result = subprocess.run(
            ["git", "-C", git_repo, "diff", "--cached", "--name-only"],
            capture_output=True, text=True, check=True,
        )
        assert "settings.py" not in result.stdout

    def test_clean_file_commits(self, git_repo):
        mgr = AutoCommitManager(scan_for_secrets=True, conventional_messages=False)
        assert mgr.enable(git_repo)
        clean_path = os.path.join(git_repo, "app.py")
        with open(clean_path, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        record = mgr.on_file_edit(clean_path, description="add helper")
        assert record is not None
        assert mgr.last_blocked_findings == []

    def test_scan_can_be_disabled(self, git_repo):
        """With scan_for_secrets=False, the scanner never runs."""
        mgr = AutoCommitManager(scan_for_secrets=False, conventional_messages=False)
        assert mgr.enable(git_repo)
        p = os.path.join(git_repo, "secret.py")
        with open(p, "w") as f:
            f.write("TOK = 'AKIAABCDEFGHIJKLMNOP'\n")
        record = mgr.on_file_edit(p, description="risky")
        assert record is not None   # went through despite the secret


class TestConventionalMessageIntegration:
    """When conventional_messages=True, the commit message carries a type."""

    def test_prefix_includes_type(self, git_repo):
        mgr = AutoCommitManager(scan_for_secrets=False, conventional_messages=True)
        assert mgr.enable(git_repo)
        tests_dir = os.path.join(git_repo, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        p = os.path.join(tests_dir, "test_x.py")
        with open(p, "w") as f:
            f.write("def test_x():\n    assert True\n")
        record = mgr.on_file_edit(p, description="add test for x")
        assert record is not None
        # Message should pick up the `test:` type from the path.
        assert "test:" in record.message

    def test_disabled_keeps_plain_message(self, git_repo):
        mgr = AutoCommitManager(scan_for_secrets=False, conventional_messages=False)
        assert mgr.enable(git_repo)
        p = os.path.join(git_repo, "m.py")
        with open(p, "w") as f:
            f.write("x = 1\n")
        record = mgr.on_file_edit(p, description="new module")
        assert record is not None
        assert "feat:" not in record.message
        assert "refactor:" not in record.message
