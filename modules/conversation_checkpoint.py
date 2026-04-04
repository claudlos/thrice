"""
Conversation Checkpointing (#7)

Implements checkpoint save/restore for long agentic sessions, allowing recovery
from failures and resumption of interrupted work. Stores conversation state
(messages, plan, tool history, metadata) in SQLite with zlib compression.

Supports both full-state checkpoints and incremental diff checkpoints for
space efficiency in long-running sessions.

Classes:
    ConversationState - Full state of a conversation at a point in time
    CheckpointInfo - Lightweight metadata about a stored checkpoint
    CheckpointManager - Save, load, list, delete checkpoints (SQLite-backed)
    DiffCheckpoint - Incremental diff-based checkpoint storage

Functions:
    serialize_state - Serialize ConversationState to compressed bytes
    deserialize_state - Deserialize compressed bytes to ConversationState
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Full state of a conversation at a point in time."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[str] = None
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "session_id": "",
        "timestamp": 0.0,
        "model": "",
        "total_tokens": 0,
        "iteration_count": 0,
    })

    def copy(self) -> ConversationState:
        """Return a deep copy of this state."""
        return deserialize_state(serialize_state(self))


@dataclass
class CheckpointInfo:
    """Lightweight metadata about a stored checkpoint."""
    checkpoint_id: str = ""
    session_id: str = ""
    label: Optional[str] = None
    timestamp: float = 0.0
    message_count: int = 0
    tool_call_count: int = 0
    size_bytes: int = 0


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_state(state: ConversationState) -> bytes:
    """Serialize a ConversationState to compressed bytes (JSON + zlib)."""
    data = json.dumps(asdict(state), separators=(",", ":"), sort_keys=True)
    return zlib.compress(data.encode("utf-8"), level=6)


def deserialize_state(data: bytes) -> ConversationState:
    """Deserialize compressed bytes back to a ConversationState."""
    raw = zlib.decompress(data)
    obj = json.loads(raw.decode("utf-8"))
    return ConversationState(
        messages=obj.get("messages", []),
        plan=obj.get("plan"),
        tool_history=obj.get("tool_history", []),
        metadata=obj.get("metadata", {}),
    )


def _generate_checkpoint_id() -> str:
    """Generate a unique checkpoint ID."""
    return uuid.uuid4().hex[:16]


def _compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def _compute_diff(old_state: ConversationState, new_state: ConversationState) -> Dict[str, Any]:
    """Compute an incremental diff between two conversation states.

    Returns a dict describing what changed. Only new messages and tool_history
    entries (appended after the old state) are stored, plus any changed
    plan/metadata.
    """
    diff: Dict[str, Any] = {"type": "diff"}

    # Messages: assume append-only; store only new messages
    old_msg_count = len(old_state.messages)
    new_msg_count = len(new_state.messages)
    if new_msg_count > old_msg_count:
        diff["new_messages"] = new_state.messages[old_msg_count:]
    elif new_msg_count < old_msg_count:
        # Truncation happened — store full messages
        diff["full_messages"] = new_state.messages
    diff["old_message_count"] = old_msg_count

    # Tool history: append-only
    old_tool_count = len(old_state.tool_history)
    new_tool_count = len(new_state.tool_history)
    if new_tool_count > old_tool_count:
        diff["new_tool_history"] = new_state.tool_history[old_tool_count:]
    elif new_tool_count < old_tool_count:
        diff["full_tool_history"] = new_state.tool_history
    diff["old_tool_count"] = old_tool_count

    # Plan
    if new_state.plan != old_state.plan:
        diff["plan"] = new_state.plan

    # Metadata: always store full (it's small)
    diff["metadata"] = new_state.metadata

    return diff


def _apply_diff(base_state: ConversationState, diff: Dict[str, Any]) -> ConversationState:
    """Apply a diff to a base state to produce the new state."""
    # Messages
    if "full_messages" in diff:
        messages = diff["full_messages"]
    else:
        messages = list(base_state.messages)
        if "new_messages" in diff:
            messages.extend(diff["new_messages"])

    # Tool history
    if "full_tool_history" in diff:
        tool_history = diff["full_tool_history"]
    else:
        tool_history = list(base_state.tool_history)
        if "new_tool_history" in diff:
            tool_history.extend(diff["new_tool_history"])

    # Plan
    plan = diff.get("plan", base_state.plan)

    # Metadata
    metadata = diff.get("metadata", base_state.metadata)

    return ConversationState(
        messages=messages,
        plan=plan,
        tool_history=tool_history,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save, load, list, and delete conversation checkpoints.

    Uses SQLite for storage with zlib-compressed state data.
    Supports both full checkpoints and incremental diff checkpoints.
    """

    DEFAULT_DB_DIR = os.path.expanduser("~/.hermes/checkpoints")

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_diff: bool = False,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.enable_diff = enable_diff
        if db_path is None:
            os.makedirs(self.DEFAULT_DB_DIR, exist_ok=True)
            self.db_path = os.path.join(self.DEFAULT_DB_DIR, f"{self.session_id}.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._ensure_db()
        self._auto_checkpoint_counter: int = 0

    def _ensure_db(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    label TEXT,
                    timestamp REAL NOT NULL,
                    message_count INTEGER NOT NULL,
                    tool_call_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    data_hash TEXT NOT NULL,
                    is_diff INTEGER NOT NULL DEFAULT 0,
                    parent_id TEXT,
                    FOREIGN KEY (parent_id) REFERENCES checkpoints(checkpoint_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON checkpoints(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON checkpoints(timestamp)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def save(
        self,
        state: ConversationState,
        label: Optional[str] = None,
    ) -> str:
        """Save a checkpoint of the current conversation state.

        Returns the checkpoint_id.
        """
        checkpoint_id = _generate_checkpoint_id()
        session_id = state.metadata.get("session_id", self.session_id)
        if not session_id:
            session_id = self.session_id
        timestamp = time.time()

        is_diff = 0
        parent_id = None

        if self.enable_diff:
            # Try to find the latest checkpoint for this session to diff against
            parent = self._get_latest_checkpoint_id(session_id)
            if parent is not None:
                parent_id = parent
                parent_state = self.load(parent_id)
                diff = _compute_diff(parent_state, state)
                data = zlib.compress(
                    json.dumps(diff, separators=(",", ":"), sort_keys=True).encode("utf-8"),
                    level=6,
                )
                is_diff = 1
            else:
                data = serialize_state(state)
        else:
            data = serialize_state(state)

        data_hash = _compute_hash(data)
        size_bytes = len(data)

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO checkpoints
                   (checkpoint_id, session_id, label, timestamp,
                    message_count, tool_call_count, size_bytes,
                    data, data_hash, is_diff, parent_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    session_id,
                    label,
                    timestamp,
                    len(state.messages),
                    len(state.tool_history),
                    size_bytes,
                    data,
                    data_hash,
                    is_diff,
                    parent_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return checkpoint_id

    def load(self, checkpoint_id: str) -> ConversationState:
        """Load a conversation state from a checkpoint.

        For diff checkpoints, recursively resolves the chain back to a
        full checkpoint.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT data, is_diff, parent_id FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        data, is_diff, parent_id = row

        if not is_diff:
            return deserialize_state(data)

        # Diff checkpoint: resolve parent chain
        # Iterative loading with depth limit to prevent unbounded recursion
        max_depth = 100
        diff_chain = []
        current_data, current_is_diff, current_parent_id = data, is_diff, parent_id
        current_id = checkpoint_id
        depth = 0
        while current_is_diff:
            if current_parent_id is None:
                raise ValueError(f"Diff checkpoint {current_id} has no parent")
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    f"Diff chain exceeds max depth ({max_depth}) at checkpoint {current_id}"
                )
            diff_raw = zlib.decompress(current_data)
            diff_chain.append(json.loads(diff_raw.decode("utf-8")))
            # Load the parent
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT data, is_diff, parent_id FROM checkpoints WHERE checkpoint_id = ?",
                    (current_parent_id,),
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                raise KeyError(f"Checkpoint not found: {current_parent_id}")
            current_id = current_parent_id
            current_data, current_is_diff, current_parent_id = row

        # current_data is now the base (non-diff) checkpoint
        state = deserialize_state(current_data)
        # Apply diffs in reverse order (oldest first)
        for diff in reversed(diff_chain):
            state = _apply_diff(state, diff)
        return state

    def list_checkpoints(self, session_id: Optional[str] = None) -> List[CheckpointInfo]:
        """List checkpoints, optionally filtered by session_id."""
        conn = self._get_conn()
        try:
            if session_id:
                rows = conn.execute(
                    """SELECT checkpoint_id, session_id, label, timestamp,
                              message_count, tool_call_count, size_bytes
                       FROM checkpoints
                       WHERE session_id = ?
                       ORDER BY timestamp ASC""",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT checkpoint_id, session_id, label, timestamp,
                              message_count, tool_call_count, size_bytes
                       FROM checkpoints
                       ORDER BY timestamp ASC""",
                ).fetchall()
        finally:
            conn.close()

        return [
            CheckpointInfo(
                checkpoint_id=r[0],
                session_id=r[1],
                label=r[2],
                timestamp=r[3],
                message_count=r[4],
                tool_call_count=r[5],
                size_bytes=r[6],
            )
            for r in rows
        ]

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint by ID.

        Also deletes any diff checkpoints that depend on this one
        (cascade orphan cleanup).
        """
        # Find dependents
        dependents = self._find_dependents(checkpoint_id)
        all_ids = [checkpoint_id] + dependents

        conn = self._get_conn()
        try:
            placeholders = ",".join("?" for _ in all_ids)
            conn.execute(
                f"DELETE FROM checkpoints WHERE checkpoint_id IN ({placeholders})",
                all_ids,
            )
            conn.commit()
        finally:
            conn.close()

    def _find_dependents(self, checkpoint_id: str) -> List[str]:
        """Recursively find all checkpoints that depend on the given one."""
        conn = self._get_conn()
        try:
            children = conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE parent_id = ?",
                (checkpoint_id,),
            ).fetchall()
        finally:
            conn.close()

        result = []
        for (child_id,) in children:
            result.append(child_id)
            result.extend(self._find_dependents(child_id))
        return result

    def auto_checkpoint(
        self,
        state: ConversationState,
        interval: int = 10,
    ) -> bool:
        """Automatically save a checkpoint every `interval` tool calls.

        Tracks the number of tool calls seen and saves when the count
        crosses a multiple of `interval`. Returns True if a checkpoint
        was saved.
        """
        current_count = len(state.tool_history)
        if current_count == 0:
            return False

        # Check if we've crossed a new interval boundary
        last_boundary = (self._auto_checkpoint_counter // interval) * interval
        current_boundary = (current_count // interval) * interval

        if current_boundary > last_boundary and current_boundary > 0:
            self._auto_checkpoint_counter = current_count
            self.save(state, label=f"auto-{current_count}")
            return True

        self._auto_checkpoint_counter = current_count
        return False

    def _get_latest_checkpoint_id(self, session_id: str) -> Optional[str]:
        """Get the most recent checkpoint ID for a session."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT checkpoint_id FROM checkpoints
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """Get the compressed size of a checkpoint in bytes."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT size_bytes FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")
        return row[0]

    def verify_integrity(self, checkpoint_id: str) -> bool:
        """Verify that a checkpoint's data matches its stored hash."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT data, data_hash FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")
        data, stored_hash = row
        return _compute_hash(data) == stored_hash

    def checkpoint_count(self, session_id: Optional[str] = None) -> int:
        """Count checkpoints, optionally filtered by session_id."""
        conn = self._get_conn()
        try:
            if session_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
        finally:
            conn.close()
        return row[0]

    def prune(self, session_id: str, keep_last: int = 5) -> int:
        """Remove old checkpoints for a session, keeping the N most recent.

        Returns the number of checkpoints deleted.
        """
        conn = self._get_conn()
        try:
            # Get IDs to keep
            keep_rows = conn.execute(
                """SELECT checkpoint_id FROM checkpoints
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, keep_last),
            ).fetchall()
            keep_ids = {r[0] for r in keep_rows}

            # Get all IDs for session
            all_rows = conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            all_ids = [r[0] for r in all_rows]

            delete_ids = [cid for cid in all_ids if cid not in keep_ids]
            if delete_ids:
                placeholders = ",".join("?" for _ in delete_ids)
                conn.execute(
                    f"DELETE FROM checkpoints WHERE checkpoint_id IN ({placeholders})",
                    delete_ids,
                )
                conn.commit()
        finally:
            conn.close()
        return len(delete_ids)


# ---------------------------------------------------------------------------
# DiffCheckpoint (convenience wrapper)
# ---------------------------------------------------------------------------

class DiffCheckpoint:
    """A convenience wrapper that always uses diff-based checkpointing.

    Wraps CheckpointManager with enable_diff=True and provides a simpler
    interface for incremental checkpoint management.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.manager = CheckpointManager(
            db_path=db_path,
            session_id=session_id,
            enable_diff=True,
        )
        self._last_state: Optional[ConversationState] = None

    @property
    def session_id(self) -> str:
        return self.manager.session_id

    def save(self, state: ConversationState, label: Optional[str] = None) -> str:
        """Save a diff checkpoint (or full if first)."""
        checkpoint_id = self.manager.save(state, label=label)
        self._last_state = state.copy()
        return checkpoint_id

    def load(self, checkpoint_id: str) -> ConversationState:
        """Load state from a checkpoint, resolving diff chain."""
        return self.manager.load(checkpoint_id)

    def list_checkpoints(self, session_id: Optional[str] = None) -> List[CheckpointInfo]:
        """List available checkpoints."""
        return self.manager.list_checkpoints(session_id=session_id)

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint and its dependents."""
        self.manager.delete(checkpoint_id)

    def save_diff_size(self, state: ConversationState) -> Tuple[int, int]:
        """Compare full vs diff checkpoint size without saving.

        Returns (full_size, diff_size) in bytes.
        """
        full_data = serialize_state(state)
        full_size = len(full_data)

        if self._last_state is None:
            return full_size, full_size

        diff = _compute_diff(self._last_state, state)
        diff_data = zlib.compress(
            json.dumps(diff, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            level=6,
        )
        return full_size, len(diff_data)
