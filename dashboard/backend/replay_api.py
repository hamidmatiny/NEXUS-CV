"""REST API for annotated inference session replay."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query

from dashboard.backend.recording_store import get_recording_store

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


@router.get("/sessions")
async def list_sessions(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, Any]]:
    """List recorded inference sessions stored in SQLite.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        List of session summary dicts.
    """
    sessions = get_recording_store().list_sessions(limit=limit)
    return [
        {
            "session_id": s.session_id,
            "camera_id": s.camera_id,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "frame_count": s.frame_count,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/frames")
async def list_session_frames(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return paginated frame metadata for a recorded session.

    Args:
        session_id: Session identifier.
        offset: Pagination offset.
        limit: Page size.

    Returns:
        Paginated frame list.
    """
    frames = get_recording_store().list_frames(session_id, offset=offset, limit=limit)
    if not frames and offset == 0:
        sessions = get_recording_store().list_sessions(limit=1000)
        if not any(s.session_id == session_id for s in sessions):
            raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "offset": offset, "limit": limit, "frames": frames}


@router.get("/sessions/{session_id}/frames/{frame_id}")
async def get_session_frame(session_id: str, frame_id: int) -> dict[str, Any]:
    """Return the full InferenceResponse payload for a recorded frame.

    Args:
        session_id: Session identifier.
        frame_id: Frame primary key.

    Returns:
        Full inference payload dict.
    """
    payload = get_recording_store().get_frame(session_id, frame_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    logger.debug("replay_frame_served", session_id=session_id, frame_id=frame_id)
    return payload
