"""Fresh-context subagent delegation for Hermes Agent (Thrice).

Grounded in the Advanced Context Engineering guide (humanlayer, 2025):

    *"Deploy fresh-context subagents for searching, summarizing, and
    file-discovery tasks, preventing these activities from clouding the
    main agent's context window with verbose tool outputs.  A bad line
    of research can land you with thousands of bad lines of code."*

This module is a **harness-level** primitive: it describes how to farm
out a bounded sub-task to a fresh context window and folds the result
back into the caller's context as a compact summary.

The runner is pluggable so the same API works for:

- in-process Python functions (unit-test friendly)
- a real LLM subagent (caller plugs in their SDK's ``messages.create``)
- a shell-spawned ``hermes`` subprocess

The dispatcher enforces three hard caps per sub-task:

- **token budget**: estimated input + output must stay under the cap
- **wall-clock timeout**: sub-task killed if it exceeds ``timeout_s``
- **output size**: response truncated + summarised if too large

And it provides **three canonical patterns** for when each matters:

- ``search_pattern``: grep/find/glob over a codebase; sub-agent returns
  a ranked list of file paths, main agent never sees the raw output.
- ``summarize_pattern``: read a long artifact, return N bullets.
- ``inspect_pattern``: read a file + compute a structured answer
  (e.g. "does this implement interface X?").

Pairs naturally with:

- ``cost_estimator``: pre-check that the sub-task is affordable
- ``task_scratchpad``: sub-agent results are appended as notes on a task
- ``trace_capture``: sub-agent exceptions surface through TraceRecord
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubTask:
    """One unit of delegated work."""

    kind: str                                # "search" | "summarize" | "inspect" | custom
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_input_tokens: int = 8000
    max_output_tokens: int = 1500
    timeout_s: float = 60.0
    truncate_output_chars: int = 6000

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt_preview": self.prompt[:80],
            "context_keys": list(self.context.keys()),
            "max_input": self.max_input_tokens,
            "max_output": self.max_output_tokens,
            "timeout_s": self.timeout_s,
        }


@dataclass
class SubResult:
    """The compact handback a caller folds into its own context."""

    ok: bool
    summary: str                              # the short answer / finding list
    raw: Optional[str] = None                 # full output, kept out of main context
    cost_tokens_in: int = 0
    cost_tokens_out: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None
    truncated: bool = False

    @property
    def cost_tokens(self) -> int:
        return self.cost_tokens_in + self.cost_tokens_out

    def format_for_caller(self, max_chars: int = 600) -> str:
        """Short blurb to paste back into the main agent's context."""
        head = f"[subagent:{'ok' if self.ok else 'fail'} · {self.cost_tokens} tokens · {self.duration_s:.1f}s]"
        body = (self.summary or self.error or "").strip()
        if len(body) > max_chars:
            body = body[: max_chars - 1] + "…"
        return f"{head}\n{body}"


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------

class Runner(Protocol):
    """Callable that actually does the work.  Pluggable so tests don't
    need a live LLM."""

    def __call__(self, task: SubTask) -> Tuple[str, int, int]:
        """Return ``(raw_output, tokens_in, tokens_out)``; may raise."""
        ...


InProcessRunner = Callable[[SubTask], str]
"""Convenience alias: in-process runners can ignore token accounting."""


def as_runner(fn: InProcessRunner, *, rough_tokens_per_char: float = 0.25) -> Runner:
    """Wrap a simple ``SubTask -> str`` function into a full Runner with
    rough token accounting.

    Any non-empty string counts as at least one token — tiny outputs
    (``"ok"``, ``"y"``) should still register on the cost tracker.
    """
    def _runner(task: SubTask) -> Tuple[str, int, int]:
        out = fn(task)
        tin = max(1 if task.prompt else 0, int(len(task.prompt) * rough_tokens_per_char))
        tout = max(1 if out else 0, int(len(out) * rough_tokens_per_char))
        return out, tin, tout
    return _runner


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

@dataclass
class DispatchStats:
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    timed_out: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0


class SubagentDispatcher:
    """Runs ``SubTask``s with caps, records telemetry, folds results
    back into a caller-friendly ``SubResult``.

    Thread-safe as long as the underlying runner is.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        default_max_parallel: int = 4,
        log_bare_prompts: bool = False,
    ):
        self._runner = runner
        self._pool = ThreadPoolExecutor(
            max_workers=default_max_parallel,
            thread_name_prefix="subagent",
        )
        self._log_bare_prompts = log_bare_prompts
        self._stats = DispatchStats()

    # -- Lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- Telemetry --------------------------------------------------------

    @property
    def stats(self) -> DispatchStats:
        # Return a copy so callers can't mutate the live state.
        return dataclasses.replace(self._stats)

    # -- Public dispatch --------------------------------------------------

    def run(self, task: SubTask) -> SubResult:
        """Run one subagent task synchronously."""
        self._stats.dispatched += 1

        if self._log_bare_prompts:
            logger.info("subagent dispatch: %s", task.to_log_dict())

        # Pre-check input size.
        approx_in = _rough_tokens(task.prompt)
        if approx_in > task.max_input_tokens:
            result = SubResult(
                ok=False,
                summary="",
                error=f"input tokens ({approx_in}) exceeds cap ({task.max_input_tokens})",
            )
            self._stats.failed += 1
            return result

        t0 = time.monotonic()
        future = self._pool.submit(self._runner, task)
        try:
            raw, tin, tout = future.result(timeout=task.timeout_s)
        except FuturesTimeout:
            duration = time.monotonic() - t0
            self._stats.timed_out += 1
            self._stats.failed += 1
            return SubResult(
                ok=False,
                summary="",
                error=f"subagent timed out after {task.timeout_s:.1f}s",
                duration_s=duration,
            )
        except Exception as exc:                # noqa: BLE001 - intentional
            duration = time.monotonic() - t0
            self._stats.failed += 1
            return SubResult(
                ok=False,
                summary="",
                error=f"{type(exc).__name__}: {exc!s}",
                duration_s=duration,
            )

        duration = time.monotonic() - t0

        truncated = False
        if len(raw) > task.truncate_output_chars:
            truncated = True
            raw_trunc = raw[: task.truncate_output_chars]
        else:
            raw_trunc = raw

        summary = _auto_summary(raw_trunc, task.kind)

        self._stats.succeeded += 1
        self._stats.total_tokens += tin + tout
        self._stats.total_duration_s += duration

        return SubResult(
            ok=True,
            summary=summary,
            raw=raw_trunc,
            cost_tokens_in=tin,
            cost_tokens_out=tout,
            duration_s=duration,
            truncated=truncated,
        )

    def run_many(
        self,
        tasks: Sequence[SubTask],
        *,
        max_parallel: Optional[int] = None,
    ) -> List[SubResult]:
        """Fan out independent sub-tasks.  Order of results matches ``tasks``."""
        if not tasks:
            return []
        max_p = max_parallel or self._pool._max_workers   # type: ignore[attr-defined]
        # Submit all, collect results in order.
        with ThreadPoolExecutor(max_workers=max_p) as pool:
            futures = [pool.submit(self.run, t) for t in tasks]
            return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Canonical task patterns
# ---------------------------------------------------------------------------

def search_pattern(query: str, paths: Sequence[str], **kw) -> SubTask:
    """Sub-task that returns a ranked list of matches for ``query`` over ``paths``."""
    prompt = (
        "You are a code-search subagent.  Find files and line numbers that "
        f"match: {query!r}.  Return at most 20 matches as a concise list; "
        f"do NOT include file bodies.  Paths to search: {list(paths)}."
    )
    return SubTask(kind="search", prompt=prompt, context={"paths": list(paths)}, **kw)


def summarize_pattern(artifact: str, focus: Optional[str] = None, **kw) -> SubTask:
    """Sub-task that returns an N-bullet summary of ``artifact``."""
    ask = f"Summarize in at most 5 bullets; focus on {focus}." if focus else \
          "Summarize in at most 5 bullets."
    prompt = f"{ask}\n\n---ARTIFACT---\n{artifact}\n---END---"
    return SubTask(kind="summarize", prompt=prompt, **kw)


def inspect_pattern(question: str, artifact: str, **kw) -> SubTask:
    """Sub-task that answers one structured question about ``artifact``."""
    prompt = (
        "Answer the question about the artifact below.  Return a JSON object "
        "with keys {answer: yes|no|partial, reason: string, evidence: string}.\n"
        f"Question: {question}\n\n---ARTIFACT---\n{artifact}\n---END---"
    )
    return SubTask(kind="inspect", prompt=prompt, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rough_tokens(s: str) -> int:
    """Cheap token estimate: ~4 chars/token for English prose."""
    return (len(s) + 3) // 4


def _auto_summary(raw: str, kind: str) -> str:
    """Trim the raw output into something concise for the main context.

    For ``inspect``, try to parse as JSON so callers get a structured
    return regardless of the model's verbosity.  For ``search``, keep
    bullet/line shape.  For everything else, take the first non-empty
    paragraph."""
    raw = raw.strip()
    if not raw:
        return ""
    if kind == "inspect":
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return json.dumps(obj, separators=(",", ":"))
        except Exception:
            pass
    if kind == "search":
        # Keep line-shaped output, cap at ~30 lines.
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        return "\n".join(lines[:30])
    # Default: first paragraph, cap at ~30 lines.
    paragraphs = raw.split("\n\n")
    return paragraphs[0][:1500]


__all__ = [
    "DispatchStats",
    "Runner",
    "SubResult",
    "SubTask",
    "SubagentDispatcher",
    "as_runner",
    "inspect_pattern",
    "search_pattern",
    "summarize_pattern",
]
