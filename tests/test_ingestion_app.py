"""Tests for ingestion FastAPI metrics endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ingestion.app import app


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingestion_health() -> None:
    """GET /health returns healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingestion_metrics() -> None:
    """GET /metrics returns Prometheus text format."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert "nexus_cv_yolo_inference_duration_ms" in response.text
