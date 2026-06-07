"""Convert domain objects to API Pydantic schemas."""

from __future__ import annotations

from collections import Counter

from fusion.data_types import BBox3D, Track
from ingestion.yolo_detector import Detection
from intelligence.data_types import AnomalyScore, ScenePrediction, TrajectoryPrediction
from serving.schemas import (
    AnomalyScoreSchema,
    BBox3DSchema,
    DetectionSchema,
    InferenceResponse,
    ScenePredictionSchema,
    TrackSchema,
    TrajectorySchema,
)


def detection_to_schema(det: Detection) -> DetectionSchema:
    """Convert a Detection to API schema.

    Args:
        det: Ingestion detection object.

    Returns:
        DetectionSchema instance.
    """
    return DetectionSchema(
        bbox_xyxy=det.bbox_xyxy,
        confidence=det.confidence,
        class_id=det.class_id,
        class_name=det.class_name,
        track_id=det.track_id,
    )


def bbox3d_to_schema(bbox: BBox3D) -> BBox3DSchema:
    """Convert BBox3D to API schema.

    Args:
        bbox: 3D bounding box.

    Returns:
        BBox3DSchema instance.
    """
    return BBox3DSchema(
        center_xyz=bbox.center_xyz,
        dimensions_lwh=bbox.dimensions_lwh,
        yaw_rad=bbox.yaw_rad,
        confidence=bbox.confidence,
    )


def track_to_schema(track: Track) -> TrackSchema:
    """Convert a fused Track to API schema.

    Args:
        track: Fusion track object.

    Returns:
        TrackSchema instance.
    """
    bbox_3d = bbox3d_to_schema(track.last_bbox_3d) if track.last_bbox_3d else None
    votes = dict(track.class_votes) if isinstance(track.class_votes, Counter) else track.class_votes
    return TrackSchema(
        track_id=track.track_id,
        state=track.state,
        age_frames=track.age_frames,
        modalities_seen=sorted(track.modalities_seen),
        last_bbox_2d=track.last_bbox_2d,
        last_bbox_3d=bbox_3d,
        velocity_2d=track.velocity_2d,
        class_votes=votes,
        anomaly_score=track.anomaly_score,
    )


def scene_to_schema(scene: ScenePrediction) -> ScenePredictionSchema:
    """Convert ScenePrediction to API schema.

    Args:
        scene: Intelligence scene prediction.

    Returns:
        ScenePredictionSchema instance.
    """
    return ScenePredictionSchema(
        scene_class=scene.scene_class,
        confidence=scene.confidence,
        top3=scene.top3,
    )


def anomaly_to_schema(anomaly: AnomalyScore) -> AnomalyScoreSchema:
    """Convert AnomalyScore to API schema.

    Args:
        anomaly: Intelligence anomaly score.

    Returns:
        AnomalyScoreSchema instance.
    """
    return AnomalyScoreSchema(
        track_id=anomaly.track_id,
        score=anomaly.score,
        contributing_factors=anomaly.contributing_factors,
        is_anomalous=anomaly.is_anomalous,
    )


def trajectory_to_schema(traj: TrajectoryPrediction) -> TrajectorySchema:
    """Convert TrajectoryPrediction to API schema.

    Args:
        traj: Intelligence trajectory prediction.

    Returns:
        TrajectorySchema instance.
    """
    return TrajectorySchema(
        track_id=traj.track_id,
        predicted_positions=traj.predicted_positions,
        horizon_frames=traj.horizon_frames,
        confidence=traj.confidence,
    )


def build_inference_response(
    request_id: str,
    camera_id: str,
    timestamp_ns: int,
    detections: list[Detection],
    tracks: list[Track],
    scene: ScenePrediction,
    anomalies: list[AnomalyScore],
    trajectories: list[TrajectoryPrediction],
    inference_ms: float,
    serving_ms: float,
) -> InferenceResponse:
    """Build a full InferenceResponse from pipeline outputs.

    Args:
        request_id: Correlation / request identifier.
        camera_id: Source camera identifier.
        timestamp_ns: Frame timestamp in nanoseconds.
        detections: YOLO detections.
        tracks: Fused tracks.
        scene: Scene classification.
        anomalies: Anomaly scores.
        trajectories: Trajectory predictions.
        inference_ms: Total inference latency.
        serving_ms: Total serving latency.

    Returns:
        InferenceResponse ready for JSON serialization.
    """
    return InferenceResponse(
        request_id=request_id,
        camera_id=camera_id,
        timestamp_ns=timestamp_ns,
        detections=[detection_to_schema(d) for d in detections],
        tracks=[track_to_schema(t) for t in tracks],
        scene=scene_to_schema(scene),
        anomalies=[anomaly_to_schema(a) for a in anomalies],
        trajectories=[trajectory_to_schema(t) for t in trajectories],
        inference_ms=inference_ms,
        serving_ms=serving_ms,
    )
