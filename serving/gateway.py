"""FastAPI gateway for the NEXUS-CV Ray Serve inference cluster."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from serving.converters import build_inference_response
from serving.health import collect_health
from serving.metrics import SLA_BREACH_TOTAL
from serving.middleware import CircuitBreakerMiddleware, CorrelationIDMiddleware, TimingMiddleware
from serving.schemas import InferenceRequest, InferenceResponse

logger = structlog.get_logger(__name__)

SLA_THRESHOLD_MS = 30.0

PipelineCallable = Callable[[InferenceRequest], Awaitable[Any]]

_pipeline_handle: PipelineCallable | None = None


def configure_pipeline(handle: PipelineCallable) -> None:
    """Configure the pipeline handle used by gateway routes.

    Args:
        handle: Async callable accepting InferenceRequest and returning PipelineResult.
    """
    global _pipeline_handle
    _pipeline_handle = handle


async def _run_pipeline(request: InferenceRequest) -> InferenceResponse:
    """Execute the inference pipeline and build the API response.

    Args:
        request: Validated inference request.

    Returns:
        InferenceResponse with full pipeline output.
    """
    if _pipeline_handle is None:
        raise RuntimeError("Pipeline not configured")

    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    result = await _pipeline_handle(request)
    serving_ms = (time.perf_counter() - start) * 1000.0
    inference_ms = result.inference_ms

    from serving.mlops_scheduler import get_mlops_scheduler

    get_mlops_scheduler().record_frame(result)

    if inference_ms > SLA_THRESHOLD_MS:
        SLA_BREACH_TOTAL.inc()
        logger.warning(
            "sla_breach",
            request_id=request_id,
            inference_ms=inference_ms,
            threshold_ms=SLA_THRESHOLD_MS,
        )

    scene = result.scene
    if scene is None:
        from intelligence.data_types import ScenePrediction

        scene = ScenePrediction(scene_class="unknown", confidence=0.0, top3=[("unknown", 0.0)])

    return build_inference_response(
        request_id=request_id,
        camera_id=request.camera_id,
        timestamp_ns=request.timestamp_ns,
        detections=result.detections,
        tracks=result.tracks,
        scene=scene,
        anomalies=result.anomalies,
        trajectories=result.trajectories,
        inference_ms=inference_ms,
        serving_ms=serving_ms,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI gateway application.

    Returns:
        Configured FastAPI instance with middleware and routes.
    """
    from serving.mlops_background import mlops_lifespan

    app = FastAPI(
        title="NEXUS-CV Serving Gateway",
        version="0.1.0",
        description="Distributed Ray Serve inference API for NEXUS-CV",
        lifespan=mlops_lifespan,
    )
    app.add_middleware(CircuitBreakerMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)

    @app.get("/health")
    async def health() -> Response:
        """Return aggregate health status."""
        report = collect_health()
        status_code = 503 if report.status == "unhealthy" else 200
        return Response(
            content=report.model_dump_json(),
            media_type="application/json",
            status_code=status_code,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Expose Prometheus metrics."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/v1/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest, http_request: Request) -> InferenceResponse:
        """Run single-frame inference through the full pipeline."""
        response = await _run_pipeline(request)
        if hasattr(http_request.state, "serving_ms"):
            response = response.model_copy(update={"serving_ms": http_request.state.serving_ms})
        return response

    @app.websocket("/ws/stream/{camera_id}")
    async def stream(websocket: WebSocket, camera_id: str) -> None:
        """Stream JPEG frames over WebSocket and receive inference JSON responses."""
        await websocket.accept()
        logger.info("websocket_connected", camera_id=camera_id)
        try:
            while True:
                data = await websocket.receive_bytes()
                import base64
                import time as time_mod

                request = InferenceRequest(
                    camera_id=camera_id,
                    frame_b64=base64.b64encode(data).decode("ascii"),
                    timestamp_ns=time_mod.time_ns(),
                )
                response = await _run_pipeline(request)
                await websocket.send_text(response.model_dump_json())
        except WebSocketDisconnect:
            logger.info("websocket_disconnected", camera_id=camera_id)

    return app


app = create_app()
