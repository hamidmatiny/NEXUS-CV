"""Pydantic v2 schemas for the NEXUS-CV serving API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TrackState = Literal["tentative", "confirmed", "lost", "dead"]


class DetectionSchema(BaseModel):
    """Single object detection in API responses."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"bbox_xyxy": [10.0, 20.0, 100.0, 200.0], "confidence": 0.92, "class_id": 0, "class_name": "person", "track_id": None}]})

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    track_id: int | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Ensure bbox coordinates are non-negative and well-formed."""
        x1, y1, x2, y2 = value
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must satisfy x2>=x1 and y2>=y1")
        if any(v < 0 for v in value):
            raise ValueError("bbox coordinates must be non-negative")
        return value


class BBox3DSchema(BaseModel):
    """3D bounding box schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "center_xyz": [1.0, 0.0, 10.0],
                    "dimensions_lwh": [4.0, 2.0, 1.5],
                    "yaw_rad": 0.0,
                    "confidence": 0.9,
                }
            ]
        }
    )

    center_xyz: tuple[float, float, float]
    dimensions_lwh: tuple[float, float, float]
    yaw_rad: float
    confidence: float = Field(ge=0.0, le=1.0)


class TrackSchema(BaseModel):
    """Fused multi-modal track schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "track_id": "550e8400-e29b-41d4-a716-446655440000",
                    "state": "confirmed",
                    "age_frames": 12,
                    "modalities_seen": ["camera", "lidar"],
                    "last_bbox_2d": [50.0, 50.0, 150.0, 150.0],
                    "last_bbox_3d": None,
                    "velocity_2d": [2.0, 1.0],
                    "class_votes": {"car": 12},
                    "anomaly_score": 0.0,
                }
            ]
        }
    )

    track_id: str = Field(min_length=1)
    state: TrackState
    age_frames: int = Field(ge=0)
    modalities_seen: list[str]
    last_bbox_2d: tuple[float, float, float, float] | None = None
    last_bbox_3d: BBox3DSchema | None = None
    velocity_2d: tuple[float, float]
    class_votes: dict[str, int]
    anomaly_score: float = Field(ge=0.0)


class ScenePredictionSchema(BaseModel):
    """Scene classification schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"scene_class": "highway", "confidence": 0.87, "top3": [["highway", 0.87], ["urban_street", 0.08], ["tunnel", 0.05]]}]
        }
    )

    scene_class: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    top3: list[tuple[str, float]]


class AnomalyScoreSchema(BaseModel):
    """Anomaly score schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "track_id": "550e8400-e29b-41d4-a716-446655440000",
                    "score": 0.75,
                    "contributing_factors": ["velocity_anomaly:z=3.5"],
                    "is_anomalous": True,
                }
            ]
        }
    )

    track_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    contributing_factors: list[str]
    is_anomalous: bool


class TrajectorySchema(BaseModel):
    """Trajectory prediction schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "track_id": "550e8400-e29b-41d4-a716-446655440000",
                    "predicted_positions": [[100.0, 100.0], [102.0, 101.0]],
                    "horizon_frames": 15,
                    "confidence": 0.75,
                }
            ]
        }
    )

    track_id: str = Field(min_length=1)
    predicted_positions: list[tuple[float, float]]
    horizon_frames: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)


class InferenceRequest(BaseModel):
    """Single-frame inference request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "camera_id": "cam_00",
                    "frame_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                    "timestamp_ns": 1_700_000_000_000_000_000,
                }
            ]
        }
    )

    camera_id: str = Field(min_length=1)
    frame_b64: str = Field(min_length=1)
    timestamp_ns: int = Field(gt=0)

    @field_validator("frame_b64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Ensure frame payload is non-empty base64."""
        if not value.strip():
            raise ValueError("frame_b64 must not be empty")
        return value


class InferenceResponse(BaseModel):
    """Full pipeline inference response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "660e8400-e29b-41d4-a716-446655440001",
                    "camera_id": "cam_00",
                    "timestamp_ns": 1_700_000_000_000_000_000,
                    "detections": [],
                    "tracks": [],
                    "scene": {"scene_class": "highway", "confidence": 0.87, "top3": [["highway", 0.87]]},
                    "anomalies": [],
                    "trajectories": [],
                    "inference_ms": 22.5,
                    "serving_ms": 25.1,
                }
            ]
        }
    )

    request_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    timestamp_ns: int = Field(gt=0)
    detections: list[DetectionSchema]
    tracks: list[TrackSchema]
    scene: ScenePredictionSchema
    anomalies: list[AnomalyScoreSchema]
    trajectories: list[TrajectorySchema]
    inference_ms: float = Field(ge=0.0)
    serving_ms: float = Field(ge=0.0)


class PipelineResult(BaseModel):
    """Internal pipeline result passed between deployments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    timestamp_ns: int
    frame: Any = None
    detections: list[Any] = Field(default_factory=list)
    tracks: list[Any] = Field(default_factory=list)
    scene: Any = None
    anomalies: list[Any] = Field(default_factory=list)
    trajectories: list[Any] = Field(default_factory=list)
    detection_ms: float = 0.0
    fusion_ms: float = 0.0
    intelligence_ms: float = 0.0

    @property
    def inference_ms(self) -> float:
        """Total inference time across all deployments."""
        return self.detection_ms + self.fusion_ms + self.intelligence_ms
