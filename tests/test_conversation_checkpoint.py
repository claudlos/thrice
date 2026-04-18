"""
Tests for conversation_checkpoint.py

Covers: save/load, auto-checkpoint, listing, deletion, compression,
diff mode, integrity, pruning, serialization helpers, edge cases.
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

# Ensure the new-files directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from conversation_checkpoint import (
    CheckpointManager,
    ConversationState,
    DiffCheckpoint,
    _apply_diff,
    _compute_diff,
    _compute_hash,
    _generate_checkpoint_id,
    deserialize_state,
    serialize_state,
)


def _make_state(
    n_messages: int = 3,
    n_tools: int = 2,
    session_id: str = "test-session",
    plan: str = "Do the thing",
) -> ConversationState:
    """Helper to create a ConversationState for testing."""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(n_messages)
    ]
    tool_history = [
        {"tool": f"tool_{i}", "args": {"x": i}, "result": f"ok_{i}"}
        for i in range(n_tools)
    ]
    metadata = {
        "session_id": session_id,
        "timestamp": time.time(),
        "model": "claude-sonnet-4-20250514",
        "total_tokens": 5000,
        "iteration_count": 10,
    }
    return ConversationState(
        messages=messages,
        plan=plan,
        tool_history=tool_history,
        metadata=metadata,
    )


class TestConversationState(unittest.TestCase):
    """Tests for the ConversationState dataclass."""

    def test_default_state(self):
        state = ConversationState()
        self.assertEqual(state.messages, [])
        self.assertIsNone(state.plan)
        self.assertEqual(state.tool_history, [])
        self.assertIn("session_id", state.metadata)

    def test_state_with_data(self):
        state = _make_state(n_messages=5, n_tools=3)
        self.assertEqual(len(state.messages), 5)
        self.assertEqual(len(state.tool_history), 3)
        self.assertEqual(state.plan, "Do the thing")

    def test_state_copy(self):
        state = _make_state()
        copy = state.copy()
        self.assertEqual(state.messages, copy.messages)
        self.assertEqual(state.plan, copy.plan)
        # Verify it's a deep copy
        copy.messages.append({"role": "user", "content": "extra"})
        self.assertNotEqual(len(state.messages), len(copy.messages))


class TestSerialization(unittest.TestCase):
    """Tests for serialize_state / deserialize_state."""

    def test_roundtrip(self):
        state = _make_state()
        data = serialize_state(state)
        restored = deserialize_state(data)
        self.assertEqual(state.messages, restored.messages)
        self.assertEqual(state.plan, restored.plan)
        self.assertEqual(state.tool_history, restored.tool_history)
        self.assertEqual(state.metadata["session_id"], restored.metadata["session_id"])

    def test_compression_smaller_than_raw(self):
        state = _make_state(n_messages=50, n_tools=20)
        compressed = serialize_state(state)
        raw = json.dumps({"messages": state.messages, "plan": state.plan,
                          "tool_history": state.tool_history,
                          "metadata": state.metadata}).encode("utf-8")
        self.assertLess(len(compressed), len(raw))

    def test_empty_state_roundtrip(self):
        state = ConversationState()
        data = serialize_state(state)
        restored = deserialize_state(data)
        self.assertEqual(restored.messages, [])
        self.assertIsNone(restored.plan)

    def test_large_state_roundtrip(self):
        state = _make_state(n_messages=500, n_tools=200)
        data = serialize_state(state)
        restored = deserialize_state(data)
        self.assertEqual(len(restored.messages), 500)
        self.assertEqual(len(restored.tool_history), 200)

    def test_serialize_produces_bytes(self):
        state = _make_state()
        data = serialize_state(state)
        self.assertIsInstance(data, bytes)


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager save/load/list/delete."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.mgr = CheckpointManager(
            db_path=self.db_path, session_id="test-session"
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_save_returns_id(self):
        state = _make_state()
        cid = self.mgr.save(state)
        self.assertIsInstance(cid, str)
        self.assertEqual(len(cid), 16)

    def test_save_and_load(self):
        state = _make_state()
        cid = self.mgr.save(state, label="first")
        loaded = self.mgr.load(cid)
        self.assertEqual(state.messages, loaded.messages)
        self.assertEqual(state.plan, loaded.plan)
        self.assertEqual(state.tool_history, loaded.tool_history)

    def test_load_nonexistent(self):
        with self.assertRaises(KeyError):
            self.mgr.load("nonexistent_id_123")

    def test_list_checkpoints(self):
        s1 = _make_state()
        s2 = _make_state(n_messages=10)
        self.mgr.save(s1, label="cp1")
        self.mgr.save(s2, label="cp2")
        cps = self.mgr.list_checkpoints()
        self.assertEqual(len(cps), 2)
        self.assertEqual(cps[0].label, "cp1")
        self.assertEqual(cps[1].label, "cp2")

    def test_list_checkpoints_filter_by_session(self):
        s1 = _make_state(session_id="session-a")
        s2 = _make_state(session_id="session-b")
        self.mgr.save(s1)
        self.mgr.save(s2)
        cps_a = self.mgr.list_checkpoints(session_id="session-a")
        cps_b = self.mgr.list_checkpoints(session_id="session-b")
        self.assertEqual(len(cps_a), 1)
        self.assertEqual(len(cps_b), 1)

    def test_delete_checkpoint(self):
        state = _make_state()
        cid = self.mgr.save(state)
        self.assertEqual(self.mgr.checkpoint_count(), 1)
        self.mgr.delete(cid)
        self.assertEqual(self.mgr.checkpoint_count(), 0)

    def test_delete_nonexistent_no_error(self):
        # Should not raise
        self.mgr.delete("nonexistent")

    def test_checkpoint_info_fields(self):
        state = _make_state(n_messages=7, n_tools=4)
        cid = self.mgr.save(state, label="test-label")
        cps = self.mgr.list_checkpoints()
        cp = cps[0]
        self.assertEqual(cp.checkpoint_id, cid)
        self.assertEqual(cp.label, "test-label")
        self.assertEqual(cp.message_count, 7)
        self.assertEqual(cp.tool_call_count, 4)
        self.assertGreater(cp.size_bytes, 0)
        self.assertGreater(cp.timestamp, 0)

    def test_verify_integrity(self):
        state = _make_state()
        cid = self.mgr.save(state)
        self.assertTrue(self.mgr.verify_integrity(cid))

    def test_verify_integrity_corrupted(self):
        state = _make_state()
        cid = self.mgr.save(state)
        # Corrupt the data
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE checkpoints SET data = ? WHERE checkpoint_id = ?",
            (b"corrupted", cid),
        )
        conn.commit()
        conn.close()
        self.assertFalse(self.mgr.verify_integrity(cid))

    def test_checkpoint_count(self):
        self.assertEqual(self.mgr.checkpoint_count(), 0)
        for i in range(5):
            self.mgr.save(_make_state())
        self.assertEqual(self.mgr.checkpoint_count(), 5)

    def test_get_checkpoint_size(self):
        state = _make_state()
        cid = self.mgr.save(state)
        size = self.mgr.get_checkpoint_size(cid)
        self.assertGreater(size, 0)

    def test_prune_keeps_last_n(self):
        for i in range(10):
            s = _make_state(session_id="prune-session")
            self.mgr.save(s, label=f"cp-{i}")
            time.sleep(0.01)  # ensure different timestamps
        deleted = self.mgr.prune("prune-session", keep_last=3)
        self.assertEqual(deleted, 7)
        remaining = self.mgr.list_checkpoints(session_id="prune-session")
        self.assertEqual(len(remaining), 3)


class TestAutoCheckpoint(unittest.TestCase):
    """Tests for auto_checkpoint."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "auto.db")
        self.mgr = CheckpointManager(
            db_path=self.db_path, session_id="auto-session"
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_auto_checkpoint_at_interval(self):
        state = _make_state(n_tools=10)
        saved = self.mgr.auto_checkpoint(state, interval=10)
        self.assertTrue(saved)

    def test_auto_checkpoint_below_interval(self):
        state = _make_state(n_tools=5)
        saved = self.mgr.auto_checkpoint(state, interval=10)
        self.assertFalse(saved)

    def test_auto_checkpoint_empty_tools(self):
        state = _make_state(n_tools=0)
        saved = self.mgr.auto_checkpoint(state, interval=10)
        self.assertFalse(saved)

    def test_auto_checkpoint_multiple_intervals(self):
        # Simulate growing tool history
        state = _make_state(n_tools=10)
        self.mgr.auto_checkpoint(state, interval=5)

        state2 = _make_state(n_tools=15)
        saved = self.mgr.auto_checkpoint(state2, interval=5)
        self.assertTrue(saved)

    def test_auto_checkpoint_creates_labeled_checkpoint(self):
        state = _make_state(n_tools=10)
        self.mgr.auto_checkpoint(state, interval=10)
        cps = self.mgr.list_checkpoints()
        self.assertEqual(len(cps), 1)
        self.assertEqual(cps[0].label, "auto-10")


class TestDiffCheckpoint(unittest.TestCase):
    """Tests for diff-based checkpointing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "diff.db")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_diff_save_and_load(self):
        dc = DiffCheckpoint(db_path=self.db_path, session_id="diff-session")
        s1 = _make_state(n_messages=5, n_tools=3, session_id="diff-session")
        cid1 = dc.save(s1, label="base")

        # Add more messages
        s2 = _make_state(n_messages=5, n_tools=3, session_id="diff-session")
        s2.messages.extend([
            {"role": "user", "content": "New message 1"},
            {"role": "assistant", "content": "New message 2"},
        ])
        s2.tool_history.append({"tool": "extra", "args": {}, "result": "ok"})
        s2.plan = "Updated plan"
        cid2 = dc.save(s2, label="diff1")

        # Load and verify
        loaded = dc.load(cid2)
        self.assertEqual(len(loaded.messages), 7)
        self.assertEqual(len(loaded.tool_history), 4)
        self.assertEqual(loaded.plan, "Updated plan")

    def test_diff_smaller_than_full(self):
        dc = DiffCheckpoint(db_path=self.db_path, session_id="diff-session")
        # Create a large base state
        s1 = _make_state(n_messages=100, n_tools=50, session_id="diff-session")
        dc.save(s1)

        # Small increment
        s2 = _make_state(n_messages=100, n_tools=50, session_id="diff-session")
        s2.messages.append({"role": "user", "content": "One more"})

        full_size, diff_size = dc.save_diff_size(s2)
        self.assertLess(diff_size, full_size)

    def test_diff_first_checkpoint_is_full(self):
        dc = DiffCheckpoint(db_path=self.db_path, session_id="diff-session")
        s1 = _make_state(session_id="diff-session")
        cid = dc.save(s1)
        loaded = dc.load(cid)
        self.assertEqual(loaded.messages, s1.messages)

    def test_diff_chain_resolution(self):
        """Test that a chain of diffs resolves correctly."""
        dc = DiffCheckpoint(db_path=self.db_path, session_id="chain-session")
        s = _make_state(n_messages=2, n_tools=1, session_id="chain-session")
        dc.save(s, label="base")

        # Build a chain of 5 diffs
        ids = []
        for i in range(5):
            s.messages.append({"role": "user", "content": f"Chain msg {i}"})
            s.tool_history.append({"tool": f"chain_{i}", "args": {}, "result": "ok"})
            cid = dc.save(s, label=f"chain-{i}")
            ids.append(cid)

        # Load the last one
        loaded = dc.load(ids[-1])
        self.assertEqual(len(loaded.messages), 7)  # 2 + 5
        self.assertEqual(len(loaded.tool_history), 6)  # 1 + 5

    def test_delete_cascades_dependents(self):
        mgr = CheckpointManager(
            db_path=self.db_path, session_id="cascade-session", enable_diff=True
        )
        s1 = _make_state(session_id="cascade-session")
        cid1 = mgr.save(s1)

        s2 = _make_state(session_id="cascade-session")
        s2.messages.append({"role": "user", "content": "extra"})
        cid2 = mgr.save(s2)

        # Delete the base should also delete the diff
        mgr.delete(cid1)
        self.assertEqual(mgr.checkpoint_count(), 0)


class TestDiffUtilities(unittest.TestCase):
    """Tests for _compute_diff and _apply_diff."""

    def test_compute_diff_append_only(self):
        old = _make_state(n_messages=3, n_tools=2)
        new = _make_state(n_messages=3, n_tools=2)
        new.messages.append({"role": "user", "content": "new"})
        new.tool_history.append({"tool": "new_tool", "args": {}, "result": "ok"})

        diff = _compute_diff(old, new)
        self.assertEqual(len(diff["new_messages"]), 1)
        self.assertEqual(len(diff["new_tool_history"]), 1)

    def test_apply_diff_restores_state(self):
        old = _make_state(n_messages=3, n_tools=2)
        new = _make_state(n_messages=3, n_tools=2)
        new.messages.append({"role": "user", "content": "new"})
        new.plan = "New plan"

        diff = _compute_diff(old, new)
        restored = _apply_diff(old, diff)
        self.assertEqual(len(restored.messages), 4)
        self.assertEqual(restored.plan, "New plan")

    def test_diff_with_truncated_messages(self):
        old = _make_state(n_messages=5, n_tools=3)
        new = _make_state(n_messages=2, n_tools=3)

        diff = _compute_diff(old, new)
        self.assertIn("full_messages", diff)
        restored = _apply_diff(old, diff)
        self.assertEqual(len(restored.messages), 2)


class TestHelpers(unittest.TestCase):
    """Tests for helper functions."""

    def test_generate_checkpoint_id_unique(self):
        ids = {_generate_checkpoint_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_compute_hash_deterministic(self):
        data = b"hello world"
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        self.assertEqual(h1, h2)

    def test_compute_hash_different_for_different_data(self):
        h1 = _compute_hash(b"hello")
        h2 = _compute_hash(b"world")
        self.assertNotEqual(h1, h2)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "edge.db")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_save_state_with_none_plan(self):
        mgr = CheckpointManager(db_path=self.db_path)
        state = ConversationState(
            messages=[{"role": "user", "content": "hi"}],
            plan=None,
            metadata={"session_id": "s1", "timestamp": 0, "model": "", "total_tokens": 0, "iteration_count": 0},
        )
        cid = mgr.save(state)
        loaded = mgr.load(cid)
        self.assertIsNone(loaded.plan)

    def test_save_state_with_unicode(self):
        mgr = CheckpointManager(db_path=self.db_path)
        state = ConversationState(
            messages=[{"role": "user", "content": "こんにちは 🌍 Ñoño"}],
            metadata={"session_id": "unicode", "timestamp": 0, "model": "", "total_tokens": 0, "iteration_count": 0},
        )
        cid = mgr.save(state)
        loaded = mgr.load(cid)
        self.assertEqual(loaded.messages[0]["content"], "こんにちは 🌍 Ñoño")

    def test_multiple_managers_same_db(self):
        mgr1 = CheckpointManager(db_path=self.db_path, session_id="s1")
        mgr2 = CheckpointManager(db_path=self.db_path, session_id="s2")
        s1 = _make_state(session_id="s1")
        s2 = _make_state(session_id="s2")
        mgr1.save(s1)
        mgr2.save(s2)
        # Both should see all checkpoints
        self.assertEqual(mgr1.checkpoint_count(), 2)
        self.assertEqual(len(mgr1.list_checkpoints(session_id="s1")), 1)
        self.assertEqual(len(mgr2.list_checkpoints(session_id="s2")), 1)


if __name__ == "__main__":
    unittest.main()
