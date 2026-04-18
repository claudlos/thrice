"""Task recitation scratchpad for Hermes Agent (Thrice).

Implements the **recitation pattern** described in the Manus context-
engineering blog (2024):

    *"The agent constantly rewrites its own todo list, pushing the
    objective into the most recent portion of the context.  This keeps
    the goal salient and directly combats lost-in-the-middle
    degradation over long (50+ step) tool-use loops."*

A ``TaskScratchpad`` holds a short, hierarchical todo list that the
agent loop re-appends to every prompt.  The list is:

- **append-only in history**: every edit writes a new revision, not a
  replace, so ``CacheTracker``/``PrefixGuard`` still see a stable
  prefix even though the *rendered* view at the end of the prompt
  changes.
- **bounded**: caps on item count, depth, and per-item length keep the
  scratchpad from ballooning into a context hog.
- **persistent**: can checkpoint to / load from a file so long runs
  survive context compaction (pairs well with ``conversation_checkpoint``).

Typical usage::

    from task_scratchpad import TaskScratchpad

    pad = TaskScratchpad(goal="migrate cron tests to async")
    pad.add("read tests/test_cron.py")
    pad.add("port fixtures to pytest-asyncio")
    pad.add("run pytest -k cron")

    prompt = (
        SYSTEM_PROMPT
        + USER_MESSAGE
        + pad.render()        # <-- goes at the END of the prompt
    )

    # As work progresses, the agent updates the pad in-place:
    pad.mark_done("read tests/test_cron.py")
    pad.add_sub("port fixtures to pytest-asyncio", "switch `setup_method` to `setup_method(self)`")
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

Status = str  # "todo" | "doing" | "done" | "blocked" | "skipped"


@dataclass
class TaskItem:
    """One entry on the scratchpad."""

    text: str
    status: Status = "todo"
    children: List["TaskItem"] = field(default_factory=list)
    note: Optional[str] = None

    def walk(self) -> Iterable["TaskItem"]:
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass(frozen=True)
class Revision:
    """Audit-log entry: what changed on the pad and when."""

    ts: float
    action: str            # "add" | "mark" | "note" | "remove" | "reset"
    path: str              # dotted path to the item, e.g. "2.1"
    detail: str


# ---------------------------------------------------------------------------
# Rendering configuration
# ---------------------------------------------------------------------------

_STATUS_GLYPH = {
    "todo":    "[ ]",
    "doing":   "[~]",
    "done":    "[x]",
    "blocked": "[!]",
    "skipped": "[-]",
}


@dataclass
class RenderConfig:
    """How the pad is formatted into prompt text."""

    header: str = "## Task scratchpad"
    include_goal: bool = True
    include_note: bool = True
    max_items: int = 20
    max_depth: int = 2
    max_text: int = 120
    compact_done: bool = True   # collapse completed items into a count


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------

class TaskScratchpad:
    """Bounded, revisable todo list for recitation at the tail of prompts."""

    def __init__(
        self,
        goal: Optional[str] = None,
        *,
        render: Optional[RenderConfig] = None,
    ):
        self._goal = goal
        self._items: List[TaskItem] = []
        self._revisions: List[Revision] = []
        self._render = render or RenderConfig()

    # -- Read --------------------------------------------------------------

    @property
    def goal(self) -> Optional[str]:
        return self._goal

    @property
    def revisions(self) -> Sequence[Revision]:
        return list(self._revisions)

    def items(self) -> Sequence[TaskItem]:
        return list(self._items)

    def __len__(self) -> int:
        return sum(1 for _ in self._walk())

    def open_count(self) -> int:
        return sum(
            1 for i in self._walk() if i.status in ("todo", "doing", "blocked")
        )

    def done_count(self) -> int:
        return sum(1 for i in self._walk() if i.status == "done")

    # -- Write -------------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        self._goal = goal
        self._log("goal", "", goal)

    def add(self, text: str, *, status: Status = "todo") -> TaskItem:
        text = self._cap(text)
        item = TaskItem(text=text, status=status)
        self._items.append(item)
        self._log("add", f"{len(self._items) - 1}", text)
        return item

    def add_sub(self, parent_text: str, text: str, *, status: Status = "todo") -> TaskItem:
        parent = self._find(parent_text)
        if parent is None:
            raise KeyError(f"parent task not found: {parent_text!r}")
        text = self._cap(text)
        item = TaskItem(text=text, status=status)
        parent.children.append(item)
        self._log("add", self._path_of(parent) + f".{len(parent.children) - 1}", text)
        return item

    def mark(self, text: str, status: Status) -> TaskItem:
        item = self._find(text)
        if item is None:
            raise KeyError(f"task not found: {text!r}")
        item.status = status
        self._log("mark", self._path_of(item), status)
        return item

    def mark_done(self, text: str) -> TaskItem:
        return self.mark(text, "done")

    def mark_doing(self, text: str) -> TaskItem:
        return self.mark(text, "doing")

    def mark_blocked(self, text: str, reason: Optional[str] = None) -> TaskItem:
        item = self.mark(text, "blocked")
        if reason:
            item.note = reason
        return item

    def note(self, text: str, note: str) -> TaskItem:
        item = self._find(text)
        if item is None:
            raise KeyError(f"task not found: {text!r}")
        item.note = self._cap(note, limit=self._render.max_text * 2)
        self._log("note", self._path_of(item), note)
        return item

    def remove(self, text: str) -> TaskItem:
        item = self._find(text)
        if item is None:
            raise KeyError(f"task not found: {text!r}")
        parent = self._parent_of(item)
        if parent is None:
            self._items.remove(item)
        else:
            parent.children.remove(item)
        self._log("remove", "?", text)
        return item

    def reset(self) -> None:
        self._items = []
        self._log("reset", "", "")

    # -- Persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "goal": self._goal,
            "items": [_item_to_dict(i) for i in self._items],
            "revisions": [dataclasses.asdict(r) for r in self._revisions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskScratchpad":
        pad = cls(goal=d.get("goal"))
        pad._items = [_dict_to_item(x) for x in d.get("items", [])]
        pad._revisions = [Revision(**r) for r in d.get("revisions", [])]
        return pad

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TaskScratchpad":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- Rendering --------------------------------------------------------

    def render(self, *, config: Optional[RenderConfig] = None) -> str:
        """Produce the compact todo-list block to append to the prompt."""
        cfg = config or self._render
        lines: List[str] = [cfg.header]

        if cfg.include_goal and self._goal:
            lines.append(f"**Goal:** {self._goal}")

        emitted = 0
        done_collapsed = 0
        # Two-pass: collect entries with depth, then emit with caps.
        for item, depth in self._walk_with_depth():
            if depth > cfg.max_depth:
                continue
            if cfg.compact_done and item.status == "done" and depth == 0:
                done_collapsed += 1
                continue
            if emitted >= cfg.max_items:
                lines.append(f"  … (+{len(self) - emitted} more)")
                break
            prefix = "  " * depth
            glyph = _STATUS_GLYPH.get(item.status, "[?]")
            text = item.text
            if len(text) > cfg.max_text:
                text = text[: cfg.max_text - 1] + "…"
            line = f"{prefix}{glyph} {text}"
            if cfg.include_note and item.note:
                note = item.note
                if len(note) > cfg.max_text:
                    note = note[: cfg.max_text - 1] + "…"
                line += f"  — {note}"
            lines.append(line)
            emitted += 1

        if cfg.compact_done and done_collapsed:
            lines.append(f"  [x] … ({done_collapsed} completed, collapsed)")

        return "\n".join(lines) + "\n"

    # -- Internals --------------------------------------------------------

    def _cap(self, text: str, *, limit: Optional[int] = None) -> str:
        limit = limit or self._render.max_text
        text = text.strip()
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return text

    def _find(self, text: str) -> Optional[TaskItem]:
        return next((i for i in self._walk() if i.text == text), None)

    def _walk(self) -> Iterable[TaskItem]:
        for root in self._items:
            yield from root.walk()

    def _walk_with_depth(self, start: Optional[Sequence[TaskItem]] = None,
                         depth: int = 0) -> Iterable:
        roots = self._items if start is None else start
        for item in roots:
            yield item, depth
            if item.children:
                yield from self._walk_with_depth(item.children, depth + 1)

    def _parent_of(self, item: TaskItem) -> Optional[TaskItem]:
        for candidate in self._walk():
            if item in candidate.children:
                return candidate
        return None

    def _path_of(self, item: TaskItem) -> str:
        # Walk top-down keeping indices.
        def find(items: List[TaskItem], trail: List[int]) -> Optional[List[int]]:
            for i, n in enumerate(items):
                if n is item:
                    return trail + [i]
                found = find(n.children, trail + [i])
                if found is not None:
                    return found
            return None
        trail = find(self._items, [])
        return ".".join(map(str, trail)) if trail else "?"

    def _log(self, action: str, path: str, detail: str) -> None:
        self._revisions.append(
            Revision(ts=time.time(), action=action, path=path, detail=detail)
        )


def _item_to_dict(item: TaskItem) -> dict:
    return {
        "text": item.text,
        "status": item.status,
        "note": item.note,
        "children": [_item_to_dict(c) for c in item.children],
    }


def _dict_to_item(d: dict) -> TaskItem:
    return TaskItem(
        text=d["text"],
        status=d.get("status", "todo"),
        note=d.get("note"),
        children=[_dict_to_item(c) for c in d.get("children", [])],
    )


__all__ = [
    "RenderConfig",
    "Revision",
    "TaskItem",
    "TaskScratchpad",
]
