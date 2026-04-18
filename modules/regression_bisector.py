"""Regression bisector for Hermes Agent (Thrice).

Wraps ``git bisect`` with a small explicit state machine.  Given a known-good
commit, a known-bad commit, and a test command, finds the first bad commit
via binary search.

Example::

    from regression_bisector import Bisector, BisectConfig

    b = Bisector(BisectConfig(
        repo_dir="/path/to/repo",
        good="v1.2.0",
        bad="HEAD",
        test_cmd=["pytest", "-q", "tests/test_regression.py"],
    ))
    result = b.run()
    if result.ok:
        print(f"first bad: {result.first_bad_commit}")
    else:
        print(f"bisect failed: {result.error}")

State machine (SM-3) mirrored in ``specs/tla/Bisector.tla``:

        ┌───────── idle ────────┐
        │            │          │
        │        mark_good     start
        │            ▼          │
        │        narrowing ─── testing
        │            │      (per-step)
        │            ▼
        └──── found ────► completed
                │
            aborted

"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

logger = logging.getLogger(__name__)


BisectState = Literal[
    "idle",
    "testing",
    "narrowing",
    "found",
    "completed",
    "aborted",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BisectConfig:
    """Configuration for a bisect run."""

    repo_dir: str
    good: str                               # commit-ish known to PASS
    bad: str                                # commit-ish known to FAIL
    test_cmd: Sequence[str]                 # argv run at each step
    step_timeout: float = 300.0             # per-commit test timeout (seconds)
    total_timeout: float = 3600.0           # overall deadline
    max_steps: int = 64                     # hard cap on bisect steps
    skip_rc: Optional[int] = 125            # rc that means "skip this commit"
    pass_rc: int = 0                        # rc that means "commit is good"


@dataclass
class BisectStep:
    """One (commit, outcome) step in the bisect."""

    commit: str
    outcome: Literal["good", "bad", "skip"]
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


@dataclass
class BisectResult:
    """Result of a bisect run."""

    ok: bool
    state: BisectState
    first_bad_commit: Optional[str] = None
    steps: List[BisectStep] = field(default_factory=list)
    error: Optional[str] = None
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BisectError(Exception):
    """Raised for unrecoverable bisect failures (git errors, path issues)."""


# ---------------------------------------------------------------------------
# The bisector
# ---------------------------------------------------------------------------

class Bisector:
    """Driver for ``git bisect``.

    Invariants (matches ``specs/tla/Bisector.tla``):

    - State transitions: ``idle → testing → narrowing`` (loop) → ``found``
      → ``completed``; ``aborted`` is reachable from any non-terminal state
      on fatal error or total-timeout.
    - ``first_bad_commit`` is only set once, in state ``found``.
    - Every step either narrows the range or aborts the bisect.
    - ``completed`` and ``aborted`` are absorbing terminal states.
    """

    def __init__(self, config: BisectConfig):
        self.config = config
        self._state: BisectState = "idle"
        self._steps: List[BisectStep] = []
        self._started_bisect = False

    # -- Public state properties ------------------------------------------

    @property
    def state(self) -> BisectState:
        return self._state

    @property
    def steps(self) -> List[BisectStep]:
        return list(self._steps)

    # -- Run ---------------------------------------------------------------

    def run(self) -> BisectResult:
        """Execute the bisect synchronously and return the result."""
        t0 = time.monotonic()
        try:
            self._verify_repo()
            self._ensure_clean_bisect()
            self._state = "testing"
            self._bisect_start()
            first_bad = self._bisect_loop(t0)
            self._bisect_reset()
            self._state = "completed"
            return BisectResult(
                ok=True,
                state=self._state,
                first_bad_commit=first_bad,
                steps=self._steps,
                duration_s=time.monotonic() - t0,
            )
        except BisectError as exc:
            logger.warning("[bisect] aborted: %s", exc)
            try:
                self._bisect_reset()
            except Exception:
                pass
            self._state = "aborted"
            return BisectResult(
                ok=False,
                state=self._state,
                steps=self._steps,
                error=str(exc),
                duration_s=time.monotonic() - t0,
            )

    # -- git plumbing -----------------------------------------------------

    def _git(self, *args: str, check: bool = True, input_: Optional[str] = None,
             timeout: float = 60.0) -> subprocess.CompletedProcess:
        cmd = ["git", *args]
        try:
            r = subprocess.run(
                cmd,
                cwd=self.config.repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input_,
            )
        except subprocess.TimeoutExpired as exc:
            raise BisectError(f"git {' '.join(args)} timed out after {timeout}s") from exc
        if check and r.returncode != 0:
            raise BisectError(
                f"git {' '.join(args)} failed (rc={r.returncode}): {r.stderr.strip()}"
            )
        return r

    def _verify_repo(self) -> None:
        r = self._git("rev-parse", "--is-inside-work-tree", check=False)
        if r.returncode != 0 or r.stdout.strip() != "true":
            raise BisectError(f"{self.config.repo_dir} is not a git repository")

    def _ensure_clean_bisect(self) -> None:
        """If a previous bisect was left half-done, reset it first."""
        self._git("bisect", "reset", check=False)

    def _bisect_start(self) -> None:
        self._git("bisect", "start")
        self._git("bisect", "bad", self.config.bad)
        self._git("bisect", "good", self.config.good)
        self._started_bisect = True

    def _bisect_reset(self) -> None:
        if self._started_bisect:
            self._git("bisect", "reset", check=False)
            self._started_bisect = False

    def _current_commit(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    # -- Main loop --------------------------------------------------------

    def _bisect_loop(self, t0: float) -> str:
        """Drive the bisect until it reports the first bad commit."""
        cfg = self.config
        self._state = "narrowing"
        for step in range(cfg.max_steps):
            if time.monotonic() - t0 > cfg.total_timeout:
                raise BisectError(
                    f"total bisect timeout ({cfg.total_timeout:.0f}s) exceeded "
                    f"after {step} steps"
                )

            commit = self._current_commit()
            self._state = "testing"
            outcome, step_record = self._run_test(commit)
            self._steps.append(step_record)
            self._state = "narrowing"

            git_outcome = {"good": "good", "bad": "bad", "skip": "skip"}[outcome]
            result = self._git("bisect", git_outcome, check=False)
            out = (result.stdout or "") + (result.stderr or "")

            first_bad = self._extract_first_bad_commit(out)
            if first_bad:
                self._state = "found"
                return first_bad

            if result.returncode not in (0, 1):
                raise BisectError(
                    f"git bisect {git_outcome} failed: {result.stderr.strip()}"
                )

        raise BisectError(
            f"bisect did not converge in {cfg.max_steps} steps (commits tested: {len(self._steps)})"
        )

    # -- Test runner ------------------------------------------------------

    def _run_test(self, commit: str) -> tuple:
        cfg = self.config
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                list(cfg.test_cmd),
                cwd=cfg.repo_dir,
                capture_output=True,
                text=True,
                timeout=cfg.step_timeout,
            )
            stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[bisect] step timed out after {cfg.step_timeout:.0f}s"
            rc = cfg.skip_rc if cfg.skip_rc is not None else 125

        if cfg.skip_rc is not None and rc == cfg.skip_rc:
            outcome = "skip"
        elif rc == cfg.pass_rc:
            outcome = "good"
        else:
            outcome = "bad"

        return outcome, BisectStep(
            commit=commit,
            outcome=outcome,         # type: ignore[arg-type]
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - t0,
        )

    # -- Parsing ---------------------------------------------------------

    @staticmethod
    def _extract_first_bad_commit(output: str) -> Optional[str]:
        """``<sha> is the first bad commit`` appears verbatim in git's output."""
        for line in output.splitlines():
            line = line.strip()
            if line.endswith("is the first bad commit"):
                return line.split()[0]
        return None
