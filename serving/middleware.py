"""ASGI middleware for correlation IDs, timing, and circuit breaking."""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from serving.metrics import CIRCUIT_BREAKER_STATE, SERVING_DURATION_MS

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

CIRCUIT_WINDOW_S = 60.0
CIRCUIT_ERROR_THRESHOLD = 0.20


class _RollingWindow:
    """Tracks success/failure over a rolling time window."""

    def __init__(self, window_s: float = CIRCUIT_WINDOW_S) -> None:
        """Initialize the rolling window.

        Args:
            window_s: Window duration in seconds.
        """
        self._window_s = window_s
        self._events: deque[tuple[float, bool]] = deque()

    def record(self, success: bool) -> None:
        """Record a request outcome.

        Args:
            success: True if the request succeeded.
        """
        now = time.time()
        self._events.append((now, success))
        self._prune(now)

    def error_rate(self) -> float:
        """Compute the error rate over the current window.

        Returns:
            Error rate in [0, 1].
        """
        self._prune(time.time())
        if not self._events:
            return 0.0
        failures = sum(1 for _, ok in self._events if not ok)
        return failures / len(self._events)

    def _prune(self, now: float) -> None:
        """Remove events outside the rolling window.

        Args:
            now: Current timestamp.
        """
        while self._events and now - self._events[0][0] > self._window_s:
            self._events.popleft()


_circuit_window = _RollingWindow()
_circuit_open = False


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID on every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request with correlation ID binding.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response with X-Request-ID header.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure wall-clock serving time and expose Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request with timing instrumentation.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response with X-Serving-Ms header.
        """
        start = time.perf_counter()
        response = await call_next(request)
        serving_ms = (time.perf_counter() - start) * 1000.0
        endpoint = request.url.path
        SERVING_DURATION_MS.labels(endpoint=endpoint).observe(serving_ms)
        response.headers["X-Serving-Ms"] = f"{serving_ms:.2f}"
        request.state.serving_ms = serving_ms
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Open circuit when error rate exceeds threshold over 60 seconds."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request with circuit breaker protection.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response or 503 when circuit is open.
        """
        global _circuit_open

        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)

        if _circuit_open:
            return JSONResponse(
                status_code=503,
                content={"detail": "Circuit breaker open"},
                headers={"Retry-After": "30"},
            )

        try:
            response = await call_next(request)
            success = response.status_code < 500
            _circuit_window.record(success)
        except Exception:
            _circuit_window.record(False)
            raise

        error_rate = _circuit_window.error_rate()
        if error_rate > CIRCUIT_ERROR_THRESHOLD and not _circuit_open:
            _circuit_open = True
            CIRCUIT_BREAKER_STATE.set(1)
            logger.warning("circuit_breaker_opened", error_rate=error_rate)
        elif error_rate <= CIRCUIT_ERROR_THRESHOLD and _circuit_open:
            _circuit_open = False
            CIRCUIT_BREAKER_STATE.set(0)
            logger.info("circuit_breaker_closed", error_rate=error_rate)

        return response


def reset_circuit_breaker() -> None:
    """Reset circuit breaker state (for testing)."""
    global _circuit_open
    _circuit_open = False
    _circuit_window._events.clear()
    CIRCUIT_BREAKER_STATE.set(0)
