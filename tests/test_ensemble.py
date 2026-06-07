"""Integration tests for IntelligenceEnsemble."""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fusion.data_types import Track
from intelligence.data_types import (
    AnomalyScore,
    IntelligenceOutput,
    ScenePrediction,
    TrajectoryPrediction,
)
from intelligence.ensemble import IntelligenceEnsemble


def _confirmed_track(track_id: str = "track-001") -> Track:
    """Build a confirmed test track."""
    return Track(
        track_id=track_id,
        state="confirmed",
        age_frames=25,
        modalities_seen={"camera", "lidar"},
        last_bbox_2d=(50.0, 50.0, 150.0, 150.0),
        last_bbox_3d=None,
        velocity_2d=(2.0, 1.0),
        class_votes=Counter({"car": 25}),
    )


@pytest.mark.unit
def test_ensemble_output_structure(synthetic_frame: np.ndarray) -> None:
    """IntelligenceEnsemble.run should return a complete IntelligenceOutput."""
    mock_scene = ScenePrediction(
        scene_class="highway",
        confidence=0.85,
        top3=[("highway", 0.85), ("urban_street", 0.1), ("tunnel", 0.05)],
    )
    mock_trajectory = TrajectoryPrediction(
        track_id="track-001",
        predicted_positions=[(100.0, 100.0)] * 15,
        horizon_frames=15,
        confidence=0.75,
    )
    mock_anomaly = AnomalyScore(
        track_id="track-001",
        score=0.0,
        contributing_factors=[],
        is_anomalous=False,
    )

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = mock_scene
    mock_predictor = MagicMock()
    mock_predictor.predict_batch.return_value = [mock_trajectory]
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = mock_anomaly
    mock_scorer.advance_frame = MagicMock()

    ensemble = IntelligenceEnsemble(
        scene_classifier=mock_classifier,
        trajectory_predictor=mock_predictor,
        anomaly_scorer=mock_scorer,
    )
    tracks = [_confirmed_track()]
    output = ensemble.run(synthetic_frame, tracks, frame_id=42, camera_id="cam_01")

    assert isinstance(output, IntelligenceOutput)
    assert output.frame_id == 42
    assert output.camera_id == "cam_01"
    assert output.scene.scene_class == "highway"
    assert len(output.trajectories) == 1
    assert len(output.anomalies) == 1
    assert output.inference_total_ms >= 0.0
    mock_scorer.advance_frame.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensemble_run_async(synthetic_frame: np.ndarray) -> None:
    """Async run should produce valid output with inference timing."""
    mock_scene = ScenePrediction(
        scene_class="urban_street", confidence=0.8, top3=[("urban_street", 0.8)]
    )
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = mock_scene
    mock_predictor = MagicMock()
    mock_predictor.predict_batch.return_value = []
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = AnomalyScore(
        track_id="t1", score=0.0, contributing_factors=[], is_anomalous=False
    )
    mock_scorer.advance_frame = MagicMock()

    with (
        patch(
            "intelligence.ensemble.asyncio.to_thread", side_effect=lambda fn, *a, **kw: fn(*a, **kw)
        ),
    ):
        ensemble = IntelligenceEnsemble(
            scene_classifier=mock_classifier,
            trajectory_predictor=mock_predictor,
            anomaly_scorer=mock_scorer,
        )
        output = await ensemble.run_async(synthetic_frame, [], frame_id=1)

    assert output.scene.scene_class == "urban_street"
    assert output.inference_total_ms >= 0.0


@pytest.mark.unit
def test_ensemble_logs_inference_ms(synthetic_frame: np.ndarray) -> None:
    """Ensemble should record wall-clock inference_total_ms."""
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = ScenePrediction(
        scene_class="tunnel", confidence=0.7, top3=[("tunnel", 0.7)]
    )
    ensemble = IntelligenceEnsemble(
        scene_classifier=mock_classifier,
        trajectory_predictor=MagicMock(predict_batch=MagicMock(return_value=[])),
        anomaly_scorer=MagicMock(
            score=MagicMock(
                return_value=AnomalyScore("t", 0.0, [], False),
            ),
            advance_frame=MagicMock(),
        ),
    )
    output = ensemble.run(synthetic_frame, [])
    assert isinstance(output.inference_total_ms, float)
