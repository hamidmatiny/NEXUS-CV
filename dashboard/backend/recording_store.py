"""SQLite-backed inference session recording for dashboard replay."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    frame_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS frames (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_index);
"""


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Summary of a recorded inference session."""

    session_id: str
    camera_id: str
    started_at: float
    ended_at: float | None
    frame_count: int


class RecordingStore:
    """Persists inference outputs to SQLite for replay."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the recording store.

        Args:
            db_path: SQLite database path (defaults to settings).
        """
        settings = get_settings()
        self._db_path = db_path or settings.RECORDING_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._active_sessions: dict[str, str] = {}

    def _init_schema(self) -> None:
        """Create database tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def start_session(self, camera_id: str) -> str:
        """Start a new recording session for a camera.

        Args:
            camera_id: Source camera identifier.

        Returns:
            New session UUID.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(tz=UTC).timestamp()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, camera_id, started_at, frame_count)
                VALUES (?, ?, ?, 0)
                """,
                (session_id, camera_id, now),
            )
        self._active_sessions[camera_id] = session_id
        logger.info("recording_session_started", session_id=session_id, camera_id=camera_id)
        return session_id

    def record_frame(
        self,
        camera_id: str,
        frame_index: int,
        timestamp_ns: int,
        payload: dict[str, Any],
    ) -> None:
        """Record a single inference frame payload.

        Args:
            camera_id: Source camera identifier.
            frame_index: Monotonic frame index within session.
            timestamp_ns: Frame timestamp in nanoseconds.
            payload: Full dashboard/replay JSON payload.
        """
        session_id = self._active_sessions.get(camera_id)
        if session_id is None:
            session_id = self.start_session(camera_id)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO frames (session_id, frame_index, timestamp_ns, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, frame_index, timestamp_ns, json.dumps(payload)),
            )
            conn.execute(
                "UPDATE sessions SET frame_count = frame_count + 1 WHERE session_id = ?",
                (session_id,),
            )

    def list_sessions(self, limit: int = 100) -> list[SessionSummary]:
        """List recorded inference sessions.

        Args:
            limit: Maximum sessions to return.

        Returns:
            List of session summaries ordered by start time descending.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, camera_id, started_at, ended_at, frame_count
                FROM sessions ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                camera_id=row["camera_id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                frame_count=row["frame_count"],
            )
            for row in rows
        ]

    def list_frames(
        self,
        session_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return paginated frame metadata for a session.

        Args:
            session_id: Session identifier.
            offset: Pagination offset.
            limit: Page size.

        Returns:
            List of frame metadata dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT frame_id, frame_index, timestamp_ns
                FROM frames WHERE session_id = ?
                ORDER BY frame_index ASC LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            ).fetchall()
        return [
            {
                "frame_id": row["frame_id"],
                "frame_index": row["frame_index"],
                "timestamp_ns": row["timestamp_ns"],
            }
            for row in rows
        ]

    def get_frame(self, session_id: str, frame_id: int) -> dict[str, Any] | None:
        """Return full inference payload for a recorded frame.

        Args:
            session_id: Session identifier.
            frame_id: Frame primary key.

        Returns:
            Parsed payload dict or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM frames
                WHERE session_id = ? AND frame_id = ?
                """,
                (session_id, frame_id),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])


_store: RecordingStore | None = None
_frame_counters: dict[str, int] = {}


def get_recording_store() -> RecordingStore:
    """Return the process-wide recording store singleton."""
    global _store
    if _store is None:
        _store = RecordingStore()
    return _store


def maybe_record_inference(camera_id: str, timestamp_ns: int, payload: dict[str, Any]) -> None:
    """Record inference output when RECORDING_ENABLED is set.

    Args:
        camera_id: Source camera identifier.
        timestamp_ns: Frame timestamp.
        payload: Dashboard JSON payload.
    """
    if not get_settings().RECORDING_ENABLED:
        return
    idx = _frame_counters.get(camera_id, 0)
    _frame_counters[camera_id] = idx + 1
    get_recording_store().record_frame(camera_id, idx, timestamp_ns, payload)
