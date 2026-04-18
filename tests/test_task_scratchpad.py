"""Tests for ``task_scratchpad``."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from task_scratchpad import (                          # noqa: E402
    RenderConfig,
    TaskScratchpad,
)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestAddAndMark:
    def test_add_and_count(self):
        pad = TaskScratchpad(goal="x")
        pad.add("one")
        pad.add("two")
        assert len(pad) == 2
        assert pad.open_count() == 2
        assert pad.done_count() == 0

    def test_mark_done(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.mark_done("a")
        assert pad.done_count() == 1
        assert pad.open_count() == 0

    def test_mark_blocked_with_reason(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.mark_blocked("a", "waiting on review")
        items = pad.items()
        assert items[0].status == "blocked"
        assert items[0].note == "waiting on review"

    def test_missing_task_raises(self):
        pad = TaskScratchpad()
        with pytest.raises(KeyError):
            pad.mark_done("nope")

    def test_remove(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.add("b")
        pad.remove("a")
        assert len(pad) == 1
        assert pad.items()[0].text == "b"


class TestSubtasks:
    def test_add_sub(self):
        pad = TaskScratchpad()
        pad.add("parent")
        pad.add_sub("parent", "child-1")
        pad.add_sub("parent", "child-2")
        assert len(pad) == 3
        parent = pad.items()[0]
        assert len(parent.children) == 2
        assert parent.children[0].text == "child-1"

    def test_add_sub_missing_parent(self):
        pad = TaskScratchpad()
        with pytest.raises(KeyError):
            pad.add_sub("ghost", "child")


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

class TestCaps:
    def test_long_text_truncated(self):
        pad = TaskScratchpad(render=RenderConfig(max_text=20))
        pad.add("x" * 100)
        assert pad.items()[0].text.endswith("…")
        assert len(pad.items()[0].text) <= 20

    def test_max_items_rendered(self):
        pad = TaskScratchpad(render=RenderConfig(max_items=3, compact_done=False))
        for i in range(10):
            pad.add(f"task-{i}")
        rendered = pad.render()
        assert "(+" in rendered        # overflow marker present
        assert rendered.count("[ ]") <= 3


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_goal_included(self):
        pad = TaskScratchpad(goal="Ship feature X")
        pad.add("a")
        out = pad.render()
        assert "Goal:" in out
        assert "Ship feature X" in out

    def test_status_glyphs(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.add("b")
        pad.mark_doing("b")
        out = pad.render()
        assert "[ ] a" in out
        assert "[~] b" in out

    def test_compact_done_collapses(self):
        pad = TaskScratchpad()
        for i in range(3):
            pad.add(f"done-{i}")
            pad.mark_done(f"done-{i}")
        pad.add("open-task")
        out = pad.render()
        # The three done items get collapsed into one "… (3 completed)" line.
        assert "3 completed" in out
        assert "[ ] open-task" in out

    def test_depth_limit(self):
        pad = TaskScratchpad(
            render=RenderConfig(max_depth=1, compact_done=False),
        )
        pad.add("p")
        pad.add_sub("p", "c1")
        # Hand-add a grandchild by reaching into the tree.
        pad.items()[0].children[0].children.append(
            type(pad.items()[0])(text="gc", status="todo", children=[])
        )
        out = pad.render()
        assert "c1" in out
        assert "gc" not in out   # depth cap hides grandchild

    def test_note_appended(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.note("a", "this is why")
        out = pad.render()
        assert "this is why" in out


# ---------------------------------------------------------------------------
# Revisions / audit log
# ---------------------------------------------------------------------------

class TestRevisions:
    def test_every_mutation_recorded(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.mark_done("a")
        pad.note("a", "n")
        pad.remove("a")
        actions = [r.action for r in pad.revisions]
        assert actions == ["add", "mark", "note", "remove"]

    def test_reset_clears_items_but_keeps_log(self):
        pad = TaskScratchpad()
        pad.add("a")
        pad.reset()
        assert len(pad) == 0
        assert any(r.action == "reset" for r in pad.revisions)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_roundtrip(self, tmp_path):
        pad = TaskScratchpad(goal="g")
        pad.add("parent")
        pad.add_sub("parent", "child")
        pad.mark_done("child")

        path = str(tmp_path / "pad.json")
        pad.save(path)
        restored = TaskScratchpad.load(path)

        assert restored.goal == "g"
        assert len(restored) == 2
        assert restored.done_count() == 1

    def test_save_is_atomic(self, tmp_path):
        """save() uses a .tmp file + rename; the final path always exists."""
        path = str(tmp_path / "pad.json")
        pad = TaskScratchpad()
        pad.add("a")
        pad.save(path)
        assert os.path.exists(path)
        # No .tmp leftover on success
        assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# End-to-end: long loop simulation
# ---------------------------------------------------------------------------

class TestLongLoop:
    def test_render_bounded_over_many_ops(self):
        """Simulate a long tool-use loop; pad must stay bounded."""
        pad = TaskScratchpad(
            goal="refactor the cron SM",
            render=RenderConfig(max_items=10),
        )
        for i in range(200):
            pad.add(f"step-{i}")
            pad.mark_done(f"step-{i}")
        # Should render without blowing up.
        out = pad.render()
        assert len(out.splitlines()) < 30   # compact even with 200 items
