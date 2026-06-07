"""Unit tests for serving ASGI middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse

from serving.middleware import (
    CircuitBreakerMiddleware,
    CorrelationIDMiddleware,
    TimingMiddleware,
    reset_circuit_breaker,
)


def _build_test_app() -> FastAPI:
    """Create a minimal FastAPI app with all serving middleware."""
    app = FastAPI()
    app.add_middleware(CircuitBreakerMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/fail")
    async def fail() -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "error"})

    return app


@pytest.fixture(autouse=True)
def _reset_circuit() -> None:
    """Reset circuit breaker state before each test."""
    reset_circuit_breaker()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_correlation_id_injected() -> None:
    """CorrelationIDMiddleware adds X-Request-ID header."""
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ok")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) >= 32


@pytest.mark.unit
@pytest.mark.asyncio
async def test_correlation_id_preserved_from_client() -> None:
    """Existing X-Request-ID header is preserved."""
    app = _build_test_app()
    custom_id = "custom-request-id-12345"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ok", headers={"X-Request-ID": custom_id})
    assert response.headers["X-Request-ID"] == custom_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timing_middleware_header() -> None:
    """TimingMiddleware adds X-Serving-Ms header."""
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ok")
    assert "X-Serving-Ms" in response.headers
    assert float(response.headers["X-Serving-Ms"]) >= 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_high_error_rate() -> None:
    """Circuit breaker returns 503 when error rate exceeds 20%."""
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(10):
            await client.get("/fail")
        response = await client.get("/ok")
    assert response.status_code == 503
    assert response.headers.get("Retry-After") == "30"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_bypasses_circuit_breaker() -> None:
    """Health and metrics paths bypass the circuit breaker."""
    app = FastAPI()
    app.add_middleware(CircuitBreakerMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/fail")
    async def fail() -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "error"})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(15):
            await client.get("/fail")
        response = await client.get("/health")
    assert response.status_code == 200
