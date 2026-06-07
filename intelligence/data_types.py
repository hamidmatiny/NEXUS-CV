"""Intelligence layer data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ScenePrediction:
    """Scene classification result for a video frame.

    Attributes:
        scene_class: Predicted scene label.
        confidence: Confidence score for the top prediction.
        top3: Top three (class, confidence) pairs.
    """

    scene_class: str
    confidence: float
    top3: list[tuple[str, float]]


@dataclass(frozen=True, slots=True)
class TrajectoryPrediction:
    """Predicted future trajectory for a tracked object.

    Attributes:
        track_id: Associated track identifier.
        predicted_positions: Future (cx, cy) positions in pixel space.
        horizon_frames: Number of future frames predicted.
        confidence: Prediction confidence in [0, 1].
    """

    track_id: str
    predicted_positions: list[tuple[float, float]]
    horizon_frames: int
    confidence: float


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """Anomaly assessment for a single track.

    Attributes:
        track_id: Associated track identifier.
        score: Combined anomaly score in [0, 1].
        contributing_factors: Human-readable factor descriptions.
        is_anomalous: True when score exceeds the anomaly threshold.
    """

    track_id: str
    score: float
    contributing_factors: list[str]
    is_anomalous: bool


@dataclass(frozen=True, slots=True)
class IntelligenceOutput:
    """Aggregated intelligence predictions for one frame.

    Attributes:
        frame_id: Frame identifier.
        camera_id: Source camera identifier.
        scene: Scene classification result.
        trajectories: Trajectory predictions for confirmed tracks.
        anomalies: Anomaly scores for confirmed tracks.
        inference_total_ms: Total wall-clock inference time in milliseconds.
    """

    frame_id: int
    camera_id: str
    scene: ScenePrediction
    trajectories: list[TrajectoryPrediction] = field(default_factory=list)
    anomalies: list[AnomalyScore] = field(default_factory=list)
    inference_total_ms: float = 0.0
