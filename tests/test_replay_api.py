"""Tests for dashboard replay API and recording store."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from config.settings import Settings
from dashboard.backend.recording_store import RecordingStore
from serving.gateway import create_app
from serving.middleware import reset_circuit_breaker


@pytest.fixture(autouse=True)
def _reset_breaker() -> None:
    """Reset circuit breaker so replay tests are not affected by gateway tests."""
    reset_circuit_breaker()


@pytest.fixture
def recording_db(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Provide an isolated SQLite recording database."""
    import dashboard.backend.recording_store as recording_module

    db_path = tmp_path / "test_recordings.db"
    rec_settings = settings.model_copy(
        update={"RECORDING_ENABLED": True, "RECORDING_DB_PATH": db_path},
    )
    monkeypatch.setattr(
        "dashboard.backend.recording_store.get_settings",
        lambda: rec_settings,
    )
    recording_module._store = None
    recording_module._frame_counters.clear()
    yield db_path
    recording_module._store = None
    recording_module._frame_counters.clear()


@pytest.fixture
def populated_store(recording_db: Path) -> RecordingStore:
    """Seed a recording store with one session and two frames."""
    store = RecordingStore(db_path=recording_db)
    session_id = store.start_session("cam_00")
    for idx in range(2):
        store.record_frame(
            "cam_00",
            idx,
            1_000_000 + idx,
            {
                "frame_b64": "abc",
                "camera_id": "cam_00",
                "detections": [],
                "tracks": [],
                "session_id": session_id,
            },
        )
    return store


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_sessions(recording_db: Path, populated_store: RecordingStore) -> None:
    """GET /api/v1/replay/sessions returns recorded sessions."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/replay/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) >= 1
    assert sessions[0]["camera_id"] == "cam_00"
    assert sessions[0]["frame_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_and_get_frames(
    recording_db: Path,
    populated_store: RecordingStore,
) -> None:
    """Paginated frame list and single-frame retrieval work."""
    sessions = populated_store.list_sessions()
    session_id = sessions[0].session_id
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        list_resp = await client.get(f"/api/v1/replay/sessions/{session_id}/frames")
        assert list_resp.status_code == 200
        frames = list_resp.json()["frames"]
        assert len(frames) == 2
        frame_id = frames[0]["frame_id"]
        frame_resp = await client.get(
            f"/api/v1/replay/sessions/{session_id}/frames/{frame_id}"
        )
    assert frame_resp.status_code == 200
    payload = frame_resp.json()
    assert payload["camera_id"] == "cam_00"
    assert payload["frame_b64"] == "abc"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_not_found(recording_db: Path) -> None:
    """Unknown session returns 404 on frame list."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/replay/sessions/missing-id/frames")
    assert response.status_code == 404
