"""Ray Serve deployments for the NEXUS-CV inference pipeline."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from fusion.data_types import AlignedReading, CameraDetectionReading
from fusion.kalman_tracker import MultiObjectTracker
from fusion.lidar_simulator import LiDARSimulator
from fusion.radar_simulator import RadarSimulator
from fusion.sensor_alignment import CameraLiDARProjector
from ingestion.yolo_detector import YOLODetector
from intelligence.ensemble import IntelligenceEnsemble
from serving.schemas import InferenceRequest, PipelineResult

logger = structlog.get_logger(__name__)

try:
    from ray import serve
except ImportError:
    serve = None  # type: ignore[assignment]


def decode_frame(frame_b64: str) -> NDArray[np.uint8]:
    """Decode a base64-encoded JPEG/PNG frame to BGR numpy array.

    On decode failure, logs a warning and returns a blank 640x480 BGR frame
    so bad or mock payloads do not crash the serving gateway.

    Args:
        frame_b64: Base64-encoded image bytes.

    Returns:
        BGR image array.
    """
    fallback = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("frame_decode_failed", error=str(exc), fallback_shape=fallback.shape)
        return fallback

    if frame is None:
        logger.warning(
            "frame_decode_failed",
            reason="cv2.imdecode returned None",
            fallback_shape=fallback.shape,
        )
        return fallback
    return frame


def _ensure_serve() -> Any:
    """Return the ray.serve module or raise ImportError."""
    if serve is None:
        raise ImportError("Ray Serve is not installed")
    return serve


def _serve_deployment(cls: type[Any], **kwargs: Any) -> Any:
    """Wrap a class with Ray Serve deployment when Ray is available.

    Maps ``max_concurrent_queries`` to Ray's ``max_ongoing_requests`` API.

    Args:
        cls: Deployment class implementation.
        **kwargs: Arguments forwarded to ``serve.deployment``.

    Returns:
        Ray Deployment wrapper or the plain class when Ray is unavailable.
    """
    if "max_concurrent_queries" in kwargs:
        kwargs["max_ongoing_requests"] = kwargs.pop("max_concurrent_queries")
    if serve is None:
        return cls
    return serve.deployment(**kwargs)(cls)


class DetectionDeployment:
    """Ray Serve deployment wrapping YOLODetector."""

    def __init__(self) -> None:
        """Initialize the YOLO detector."""
        from serving.model_loader import resolve_yolo_model_path

        self._detector = YOLODetector(model_path=resolve_yolo_model_path())
        logger.info("detection_deployment_initialized")

    async def __call__(self, request: InferenceRequest) -> PipelineResult:
        """Run object detection on a single frame.

        Args:
            request: Inference request with base64-encoded frame.

        Returns:
            PipelineResult with detections populated.
        """
        start = time.perf_counter()
        frame = await asyncio.to_thread(decode_frame, request.frame_b64)
        detections = await asyncio.to_thread(self._detector.detect, frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        from serving.metrics import SERVE_INFERENCE_DURATION_MS

        SERVE_INFERENCE_DURATION_MS.labels(
            deployment="detection", camera_id=request.camera_id
        ).observe(elapsed_ms)
        return PipelineResult(
            camera_id=request.camera_id,
            timestamp_ns=request.timestamp_ns,
            frame=frame,
            detections=detections,
            detection_ms=elapsed_ms,
        )


class FusionDeployment:
    """Stateful Ray Serve deployment for multi-modal track fusion."""

    def __init__(self) -> None:
        """Initialize fusion components."""
        self._tracker = MultiObjectTracker()
        self._lidar_sim = LiDARSimulator()
        self._radar_sim = RadarSimulator()
        self._projector = CameraLiDARProjector()
        self._prev_detections: list[Any] = []
        logger.info("fusion_deployment_initialized")

    async def __call__(self, partial: PipelineResult) -> PipelineResult:
        """Fuse detections into enriched tracks.

        Args:
            partial: Pipeline result from detection stage.

        Returns:
            PipelineResult with tracks populated.
        """
        start = time.perf_counter()
        frame = partial.frame
        if frame is None:
            raise ValueError("Detection stage did not provide a decoded frame")

        detections = partial.detections
        frame_shape = frame.shape
        lidar = await asyncio.to_thread(
            self._lidar_sim.generate,
            detections,
            frame_shape,
            partial.timestamp_ns,
        )
        radar = await asyncio.to_thread(
            self._radar_sim.generate,
            detections,
            frame_shape,
            partial.timestamp_ns,
            self._prev_detections,
        )
        camera = CameraDetectionReading(
            sensor_id=partial.camera_id,
            timestamp_ns=partial.timestamp_ns,
            detections=detections,
        )
        aligned = AlignedReading(
            camera=camera,
            lidar=lidar,
            radar=radar,
            alignment_gap_ms=0.0,
        )

        mot_tracks = await asyncio.to_thread(self._tracker.update, detections)
        enriched = self._fuse_modalities(mot_tracks, aligned)
        self._prev_detections = detections

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        from serving.metrics import ACTIVE_TRACKS, SERVE_INFERENCE_DURATION_MS

        SERVE_INFERENCE_DURATION_MS.labels(
            deployment="fusion", camera_id=partial.camera_id
        ).observe(elapsed_ms)
        ACTIVE_TRACKS.set(len(enriched))

        partial.tracks = enriched
        partial.fusion_ms = elapsed_ms
        return partial

    def _fuse_modalities(self, tracks: list[Any], aligned: AlignedReading) -> list[Any]:
        """Assign LiDAR and radar modalities to confirmed tracks.

        Args:
            tracks: MOT tracker output tracks.
            aligned: Aligned multi-modal reading.

        Returns:
            Enriched track list.
        """
        from dataclasses import replace

        from fusion.fusion_actor import _nearest_lidar_cluster, _nearest_radar_target

        enriched = []
        for track in tracks:
            if track.state != "confirmed" or track.last_bbox_2d is None:
                enriched.append(track)
                continue
            modalities = set(track.modalities_seen)
            bbox_3d = track.last_bbox_3d
            velocity = track.velocity_2d
            cluster = _nearest_lidar_cluster(track.last_bbox_2d, aligned, self._projector)
            if cluster is not None:
                bbox_3d = cluster
                modalities.add("lidar")
            radar_target = _nearest_radar_target(track.last_bbox_2d, aligned)
            if radar_target is not None:
                velocity = (track.velocity_2d[0], radar_target.velocity_mps)
                modalities.add("radar")
            enriched.append(
                replace(
                    track,
                    last_bbox_3d=bbox_3d,
                    velocity_2d=velocity,
                    modalities_seen=modalities,
                )
            )
        return enriched


class IntelligenceDeployment:
    """Ray Serve deployment wrapping IntelligenceEnsemble."""

    def __init__(self) -> None:
        """Initialize the intelligence ensemble."""
        from intelligence.trajectory_lstm import TrajectoryPredictor
        from serving.model_loader import resolve_lstm_model_path

        lstm_path = resolve_lstm_model_path()
        self._ensemble = IntelligenceEnsemble(
            trajectory_predictor=TrajectoryPredictor(model_path=lstm_path),
        )
        logger.info("intelligence_deployment_initialized", lstm_path=lstm_path)

    async def __call__(self, partial: PipelineResult) -> PipelineResult:
        """Run scene classification, trajectory, and anomaly scoring.

        Args:
            partial: Pipeline result from fusion stage.

        Returns:
            PipelineResult with intelligence outputs populated.
        """
        start = time.perf_counter()
        frame = partial.frame
        if frame is None:
            raise ValueError("Pipeline missing decoded frame")

        output = await self._ensemble.run_async(
            frame,
            partial.tracks,
            frame_id=0,
            camera_id=partial.camera_id,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        from serving.metrics import ANOMALY_DETECTIONS_TOTAL, SERVE_INFERENCE_DURATION_MS

        SERVE_INFERENCE_DURATION_MS.labels(
            deployment="intelligence", camera_id=partial.camera_id
        ).observe(elapsed_ms)

        for anomaly in output.anomalies:
            if anomaly.is_anomalous:
                for factor in anomaly.contributing_factors:
                    ANOMALY_DETECTIONS_TOTAL.labels(
                        camera_id=partial.camera_id,
                        factor=factor.split(":")[0],
                    ).inc()

        partial.scene = output.scene
        partial.anomalies = output.anomalies
        partial.trajectories = output.trajectories
        partial.intelligence_ms = elapsed_ms
        return partial


def build_pipeline() -> Any:
    """Build the chained NexusCVPipeline application DAG.

    Returns:
        Ray Serve bound application graph, or LocalPipeline when Ray unavailable.
    """
    if serve is None:
        return LocalPipeline()

    detection = _serve_deployment(
        DetectionDeployment,
        name="DetectionDeployment",
        num_replicas=2,
        max_concurrent_queries=10,
        ray_actor_options={"num_cpus": 1},
    )
    fusion = _serve_deployment(
        FusionDeployment,
        name="FusionDeployment",
        num_replicas=1,
        max_concurrent_queries=5,
        ray_actor_options={"num_cpus": 1},
    )
    intelligence = _serve_deployment(
        IntelligenceDeployment,
        name="IntelligenceDeployment",
        num_replicas=2,
        max_concurrent_queries=8,
        ray_actor_options={"num_cpus": 1},
    )
    return intelligence.bind(fusion.bind(detection.bind()))


def build_serve_application() -> Any:
    """Build the full Ray Serve application with FastAPI ingress.

    Returns:
        Bound NexusCVGateway ingress over the inference pipeline DAG.
    """
    _ensure_serve()
    from serving.gateway import app, configure_pipeline

    pipeline_binding = build_pipeline()

    class _NexusCVGateway:
        """Ray Serve ingress deployment wrapping the FastAPI gateway."""

        def __init__(self, pipeline: Any) -> None:
            """Bind the pipeline handle to gateway routes.

            Args:
                pipeline: Ray Serve deployment handle.
            """

            async def _call(request: InferenceRequest) -> PipelineResult:
                return await pipeline.remote(request)

            configure_pipeline(_call)

    gateway = _serve_deployment(
        serve.ingress(app)(_NexusCVGateway),
        name="NexusCVGateway",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
    )
    return gateway.bind(pipeline_binding)


NexusCVPipeline = build_pipeline


class LocalPipeline:
    """In-process pipeline for tests and gRPC without Ray Serve."""

    def __init__(
        self,
        detection: DetectionDeployment | None = None,
        fusion: FusionDeployment | None = None,
        intelligence: IntelligenceDeployment | None = None,
    ) -> None:
        """Initialize local deployment instances.

        Args:
            detection: Optional detection deployment override.
            fusion: Optional fusion deployment override.
            intelligence: Optional intelligence deployment override.
        """
        self._detection = detection or DetectionDeployment()
        self._fusion = fusion or FusionDeployment()
        self._intelligence = intelligence or IntelligenceDeployment()

    async def remote(self, request: InferenceRequest) -> PipelineResult:
        """Run the full pipeline locally.

        Args:
            request: Inference request.

        Returns:
            Completed PipelineResult.
        """
        partial = await self._detection(request)
        partial = await self._fusion(partial)
        return await self._intelligence(partial)


_shared_pipeline: LocalPipeline | None = None


def get_shared_pipeline() -> LocalPipeline:
    """Return a process-wide LocalPipeline singleton (lazy-loaded models).

    Returns:
        Shared LocalPipeline instance.
    """
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = LocalPipeline()
    return _shared_pipeline


def reset_shared_pipeline() -> None:
    """Reset the shared pipeline singleton (for tests)."""
    global _shared_pipeline
    _shared_pipeline = None
