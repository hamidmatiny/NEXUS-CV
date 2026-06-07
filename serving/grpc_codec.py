"""Convert between Pydantic API schemas and gRPC protobuf messages."""

from __future__ import annotations

from proto import nexus_cv_pb2
from serving.schemas import InferenceResponse


def inference_response_to_proto(response: InferenceResponse) -> nexus_cv_pb2.InferenceResponse:
    """Convert an InferenceResponse to a protobuf message.

    Args:
        response: Pydantic inference response.

    Returns:
        Protobuf InferenceResponse message.
    """
    detections = [
        nexus_cv_pb2.Detection(
            bbox_xyxy=list(d.bbox_xyxy),
            confidence=d.confidence,
            class_id=d.class_id,
            class_name=d.class_name,
            track_id=d.track_id if d.track_id is not None else 0,
        )
        for d in response.detections
    ]

    tracks = []
    for track in response.tracks:
        bbox_3d = None
        if track.last_bbox_3d is not None:
            bbox_3d = nexus_cv_pb2.BBox3D(
                center_xyz=list(track.last_bbox_3d.center_xyz),
                dimensions_lwh=list(track.last_bbox_3d.dimensions_lwh),
                yaw_rad=track.last_bbox_3d.yaw_rad,
                confidence=track.last_bbox_3d.confidence,
            )
        tracks.append(
            nexus_cv_pb2.Track(
                track_id=track.track_id,
                state=track.state,
                age_frames=track.age_frames,
                modalities_seen=track.modalities_seen,
                last_bbox_2d=list(track.last_bbox_2d) if track.last_bbox_2d else [],
                last_bbox_3d=bbox_3d,
                velocity_2d=list(track.velocity_2d),
                class_votes=track.class_votes,
                anomaly_score=track.anomaly_score,
            )
        )

    scene = nexus_cv_pb2.ScenePrediction(
        scene_class=response.scene.scene_class,
        confidence=response.scene.confidence,
        top3=[
            nexus_cv_pb2.SceneClassScore(class_name=name, score=score)
            for name, score in response.scene.top3
        ],
    )

    anomalies = [
        nexus_cv_pb2.AnomalyScore(
            track_id=a.track_id,
            score=a.score,
            contributing_factors=a.contributing_factors,
            is_anomalous=a.is_anomalous,
        )
        for a in response.anomalies
    ]

    trajectories = [
        nexus_cv_pb2.Trajectory(
            track_id=t.track_id,
            predicted_positions=[
                nexus_cv_pb2.Position2D(x=x, y=y) for x, y in t.predicted_positions
            ],
            horizon_frames=t.horizon_frames,
            confidence=t.confidence,
        )
        for t in response.trajectories
    ]

    return nexus_cv_pb2.InferenceResponse(
        request_id=response.request_id,
        camera_id=response.camera_id,
        timestamp_ns=response.timestamp_ns,
        detections=detections,
        tracks=tracks,
        scene=scene,
        anomalies=anomalies,
        trajectories=trajectories,
        inference_ms=response.inference_ms,
        serving_ms=response.serving_ms,
    )


def inference_request_from_proto(request: nexus_cv_pb2.InferenceRequest) -> tuple[str, str, int]:
    """Extract fields from a protobuf InferenceRequest.

    Args:
        request: Protobuf inference request.

    Returns:
        Tuple of (camera_id, frame_b64, timestamp_ns).
    """
    return request.camera_id, request.frame_b64, request.timestamp_ns
