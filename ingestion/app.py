"""FastAPI health and Prometheus metrics endpoint for the ingestion service."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from ingestion import metrics as _ingestion_metrics  # noqa: F401 — register counters

app = FastAPI(
    title="NEXUS-CV Ingestion",
    version="0.1.0",
    description="Health and metrics for the ingestion pipeline",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return ingestion service health."""
    return {"status": "healthy", "service": "ingestion"}


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for the ingestion pipeline."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
