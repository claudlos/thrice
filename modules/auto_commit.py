"""
Auto-commit + /undo system for Hermes.

Automatically commits file changes made by the AI with descriptive messages,
and provides undo capabilities to revert AI-made changes.

Usage:
    manager = AutoCommitManager()
    manager.enable("/path/to/project")
    manager.on_file_edit("src/main.py", "Added error handling to parse function")
    # Creates commit: "hermes: Added error handling to parse function"

    undo = UndoManager("/path/to/project")
    undo.undo()       # Revert last AI commit
    undo.undo_all()   # Revert all AI commits in session
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default commit message prefix used to identify AI commits
DEFAULT_PREFIX = "hermes:"

# Branch prefix for AI work
AI_BRANCH_PREFIX = "hermes/"


@dataclass
class CommitRecord:
    """Record of an AI-made commit."""
    sha: str
    message: str
    timestamp: float
    files: List[str]


def _run_git(args: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    cmd = ["git"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out") from exc


class GitError(Exception):
    """Raised when a git operation fails."""
    pass


class AutoCommitManager:
    """
    Manages automatic git commits for AI-made file changes.

    Safety rules:
    - Never auto-commit on detached HEAD
    - Never auto-commit if there are merge conflicts
    - Never auto-commit to main/master (creates a feature branch)
    """

    def __init__(
        self,
        prefix: str = DEFAULT_PREFIX,
        *,
        scan_for_secrets: bool = True,
        conventional_messages: bool = True,
    ):
        self.prefix = prefix
        self.project_dir: Optional[str] = None
        self._enabled = False
        self._session_commits: List[CommitRecord] = []
        self._pending_files: List[Tuple[str, str]] = []  # (file_path, description)
        # Optional integrations - gracefully no-op if the modules aren't present.
        self.scan_for_secrets = scan_for_secrets
        self.conventional_messages = conventional_messages
        self._last_blocked_findings: list = []

    @property
    def last_blocked_findings(self) -> list:
        """Findings from the most recent commit attempt that was blocked by
        the secret scanner.  Empty list if the last attempt wasn't blocked."""
        return list(self._last_blocked_findings)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def session_commits(self) -> List[CommitRecord]:
        return list(self._session_commits)

    def enable(self, project_dir: str) -> bool:
        """
        Enable auto-commits for the given project directory.
        Returns True if successfully enabled, False if conditions aren't met.
        """
        project_dir = os.path.abspath(project_dir)

        # Check it's a git repo
        if not os.path.isdir(os.path.join(project_dir, ".git")):
            logger.warning("Not a git repository: %s", project_dir)
            return False

        self.project_dir = project_dir

        # Safety checks
        if self._is_detached_head():
            logger.warning("Detached HEAD detected, auto-commit disabled")
            return False

        if self._has_merge_conflicts():
            logger.warning("Merge conflicts detected, auto-commit disabled")
            return False

        # If on main/master, create a feature branch
        branch = self._current_branch()
        if branch in ("main", "master"):
            new_branch = f"{AI_BRANCH_PREFIX}session-{int(time.time())}"
            try:
                _run_git(["checkout", "-b", new_branch], self.project_dir)
                logger.info("Created feature branch: %s", new_branch)
            except GitError as e:
                logger.warning("Failed to create feature branch: %s", e)
                return False

        self._enabled = True
        self._session_commits = []
        logger.info("Auto-commit enabled for %s", project_dir)
        return True

    def disable(self):
        """Disable auto-commits. Flushes any pending changes first."""
        if self._enabled and self._pending_files:
            self.commit_if_pending()
        self._enabled = False
        logger.info("Auto-commit disabled")

    def on_file_edit(self, file_path: str, description: str) -> Optional[CommitRecord]:
        """
        Called after each successful file edit. Creates a commit immediately.
        Returns the CommitRecord if committed, None if skipped.
        """
        if not self._enabled or not self.project_dir:
            return None

        # Re-check safety
        if self._is_detached_head() or self._has_merge_conflicts():
            logger.warning("Safety check failed, skipping auto-commit")
            return None

        # Stage the file
        abs_path = os.path.abspath(file_path)
        try:
            rel_path = os.path.relpath(abs_path, self.project_dir)
        except ValueError:
            rel_path = abs_path

        try:
            _run_git(["add", rel_path], self.project_dir)
        except GitError as e:
            logger.warning("Failed to stage file %s: %s", rel_path, e)
            return None

        # Check if there are actually staged changes
        result = _run_git(["diff", "--cached", "--name-only"], self.project_dir, check=False)
        if not result.stdout.strip():
            logger.debug("No changes to commit for %s", rel_path)
            return None

        # Pre-commit secret scan.  If a high/medium-severity finding trips,
        # unstage and refuse - the agent can then fix and retry.
        if self.scan_for_secrets and not self._secret_scan_ok():
            try:
                _run_git(["reset", rel_path], self.project_dir, check=False)
            except GitError:
                pass
            return None

        # Build the commit message.  If conventional-commits integration is
        # available and enabled, promote "<prefix> <desc>" to
        # "<prefix> <type>(<scope>): <desc>".
        message = self._format_message(description, rel_path)

        try:
            _run_git(["commit", "-m", message], self.project_dir)
        except GitError as e:
            logger.warning("Failed to commit: %s", e)
            return None

        # Get the commit SHA
        result = _run_git(["rev-parse", "HEAD"], self.project_dir)
        sha = result.stdout.strip()

        record = CommitRecord(
            sha=sha,
            message=message,
            timestamp=time.time(),
            files=[rel_path],
        )
        self._session_commits.append(record)
        logger.info("Auto-committed: %s (%s)", message, sha[:8])
        return record

    def commit_if_pending(self) -> Optional[CommitRecord]:
        """Commit any pending staged changes as a batch."""
        if not self._enabled or not self.project_dir:
            return None

        # Stage all tracked modified files
        result = _run_git(["diff", "--name-only"], self.project_dir, check=False)
        if not result.stdout.strip():
            return None

        try:
            _run_git(["add", "-u"], self.project_dir)
            message = f"{self.prefix} batch update"
            _run_git(["commit", "-m", message], self.project_dir)
        except GitError as e:
            logger.warning("Failed to commit pending: %s", e)
            return None

        result = _run_git(["rev-parse", "HEAD"], self.project_dir)
        sha = result.stdout.strip()

        record = CommitRecord(
            sha=sha,
            message=message,
            timestamp=time.time(),
            files=[],
        )
        self._session_commits.append(record)
        return record

    def _current_branch(self) -> str:
        """Get current branch name."""
        result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], self.project_dir, check=False)
        return result.stdout.strip()

    def _is_detached_head(self) -> bool:
        """Check if repo is in detached HEAD state."""
        return self._current_branch() == "HEAD"

    def _has_merge_conflicts(self) -> bool:
        """Check if there are unresolved merge conflicts."""
        result = _run_git(["ls-files", "--unmerged"], self.project_dir, check=False)
        return bool(result.stdout.strip())

    # ------------------------------------------------------------------
    # Optional integrations with secret_scanner + conventional_commit
    # ------------------------------------------------------------------

    def _secret_scan_ok(self) -> bool:
        """Run ``secret_scanner.scan_diff`` on the staged changes.  Returns
        False (and logs a warning) if a high-or-medium-severity finding
        fires.  If ``secret_scanner`` isn't installed, returns True so the
        commit proceeds - graceful degradation."""
        try:
            from secret_scanner import scan_diff
        except ImportError:
            return True
        cached = _run_git(["diff", "--cached"], self.project_dir, check=False)
        findings = [
            f for f in scan_diff(cached.stdout or "")
            if f.severity in ("high", "medium")
        ]
        self._last_blocked_findings = findings
        if findings:
            logger.warning(
                "auto_commit: blocked; %d secret finding(s): %s",
                len(findings),
                ", ".join(f"{f.rule} at {f.file}:{f.line}" for f in findings[:3]),
            )
            return False
        return True

    def _format_message(self, description: str, rel_path: str) -> str:
        """Return the final commit message.  Adds a Conventional-Commits
        type prefix when ``conventional_commit`` is importable; otherwise
        falls back to ``<prefix> <description>``."""
        base = f"{self.prefix} {description}"
        if not self.conventional_messages:
            return base
        try:
            from conventional_commit import suggest_type
        except ImportError:
            return base
        try:
            cached = _run_git(["diff", "--cached"], self.project_dir, check=False)
            ctype = suggest_type(cached.stdout or "", paths=[rel_path])
        except Exception:
            return base
        desc = description.strip().rstrip(".")
        if desc and desc[0].isupper():
            desc = desc[0].lower() + desc[1:]
        return f"{self.prefix} {ctype}: {desc}"


class UndoManager:
    """
    Manages undo operations for AI-made commits.
    Identifies AI commits by the message prefix.
    """

    def __init__(self, project_dir: str, prefix: str = DEFAULT_PREFIX,
                 session_commits: Optional[List[CommitRecord]] = None):
        self.project_dir = os.path.abspath(project_dir)
        self.prefix = prefix
        self._session_commits = session_commits or []

    def undo(self) -> Optional[str]:
        """
        Revert the last AI commit.
        Returns the reverted commit SHA, or None if nothing to undo.
        """
        # Check if HEAD is an AI commit
        result = _run_git(["log", "-1", "--format=%H %s"], self.project_dir, check=False)
        if not result.stdout.strip():
            return None

        parts = result.stdout.strip().split(" ", 1)
        sha = parts[0]
        message = parts[1] if len(parts) > 1 else ""

        if not message.startswith(self.prefix):
            logger.info("HEAD is not an AI commit, nothing to undo")
            return None

        # Use git revert to safely undo
        try:
            _run_git(["revert", "--no-edit", "HEAD"], self.project_dir)
            logger.info("Reverted AI commit: %s (%s)", message, sha[:8])
            # Remove from session tracking
            self._session_commits = [c for c in self._session_commits if c.sha != sha]
            return sha
        except GitError:
            # If revert fails (conflicts), try reset — only checkout the specific files
            logger.warning("Revert failed, trying soft reset")
            try:
                # Get the files changed in this commit before resetting
                file_result = _run_git(
                    ["diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                    self.project_dir,
                    check=False,
                )
                changed_files = [f for f in file_result.stdout.strip().split("\n") if f]
                _run_git(["reset", "--soft", "HEAD~1"], self.project_dir)
                # Only checkout the specific files that were committed, not all unstaged work
                if changed_files:
                    _run_git(["checkout", "--"] + changed_files, self.project_dir)
                return sha
            except GitError as e:
                logger.error("Failed to undo: %s", e)
                return None

    def undo_all(self) -> List[str]:
        """
        Revert all AI commits in the current session (most recent first).
        Returns list of reverted commit SHAs.
        """
        reverted = []
        # Try session commits first (more precise)
        if self._session_commits:
            for commit in reversed(self._session_commits):
                # Verify this commit is still HEAD
                result = _run_git(["rev-parse", "HEAD"], self.project_dir, check=False)
                if result.stdout.strip() == commit.sha:
                    sha = self.undo()
                    if sha:
                        reverted.append(sha)
            return reverted

        # Fallback: undo consecutive AI commits from HEAD
        max_iterations = 50  # safety limit
        for _ in range(max_iterations):
            sha = self.undo()
            if sha is None:
                break
            reverted.append(sha)

        return reverted

    def get_ai_commits(self, limit: int = 50) -> List[CommitRecord]:
        """List AI commits identified by the prefix in commit messages."""
        result = _run_git(
            ["log", f"--max-count={limit}", "--format=%H|%s|%at"],
            self.project_dir,
            check=False,
        )
        if not result.stdout.strip():
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            sha, message, timestamp_str = parts
            if message.startswith(self.prefix):
                # Get files changed
                file_result = _run_git(
                    ["diff-tree", "--no-commit-id", "--name-only", "-r", sha],
                    self.project_dir,
                    check=False,
                )
                files = [f for f in file_result.stdout.strip().split("\n") if f]
                commits.append(CommitRecord(
                    sha=sha,
                    message=message,
                    timestamp=float(timestamp_str),
                    files=files,
                ))
        return commits

    def diff(self) -> str:
        """Show current uncommitted changes."""
        result = _run_git(["diff"], self.project_dir, check=False)
        staged = _run_git(["diff", "--cached"], self.project_dir, check=False)
        output = ""
        if result.stdout:
            output += "=== Unstaged Changes ===\n" + result.stdout
        if staged.stdout:
            output += "\n=== Staged Changes ===\n" + staged.stdout
        if not output:
            output = "No uncommitted changes."
        return output


def is_auto_commit_enabled() -> bool:
    """Check if auto-commit is enabled via environment or config."""
    env_val = os.environ.get("HERMES_AUTO_COMMIT", "").lower()
    if env_val in ("true", "1", "yes"):
        return True
    if env_val in ("false", "0", "no"):
        return False

    # Check config.yaml
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            return config.get("auto_commit", False)
        except Exception:
            pass

    return False
