"""Unit tests for TrajectoryLSTM and TrajectoryPredictor."""

from __future__ import annotations

from collections import Counter

import pytest

from fusion.data_types import Track
from intelligence.trajectory_lstm import HORIZON, TrajectoryLSTM, TrajectoryPredictor


def _track(
    track_id: str,
    bbox: tuple[float, float, float, float],
    velocity: tuple[float, float] = (2.0, 1.0),
    age: int = 25,
) -> Track:
    """Build a test Track with sufficient age."""
    return Track(
        track_id=track_id,
        state="confirmed",
        age_frames=age,
        modalities_seen={"camera"},
        last_bbox_2d=bbox,
        last_bbox_3d=None,
        velocity_2d=velocity,
        class_votes=Counter({"car": age}),
    )


@pytest.mark.unit
def test_forward_pass_shapes() -> None:
    """Forward pass should produce (B, horizon, 2) output."""
    pytest.importorskip("torch")
    import torch

    model = TrajectoryLSTM(horizon=HORIZON)
    model.eval()
    x = torch.randn(4, 20, 6)
    out = model(x)
    assert out.shape == (4, HORIZON, 2)


@pytest.mark.unit
def test_predict_batch_insufficient_history() -> None:
    """Tracks with fewer than seq_len observations should be skipped."""
    pytest.importorskip("torch")
    predictor = TrajectoryPredictor(model_path="/nonexistent/model.pt", seq_len=20)
    track = _track("t1", (10.0, 10.0, 60.0, 60.0), age=1)
    results = predictor.predict_batch([track])
    assert results == []


@pytest.mark.unit
def test_predict_batch_with_sufficient_history() -> None:
    """Tracks with enough history should receive trajectory predictions."""
    pytest.importorskip("torch")
    predictor = TrajectoryPredictor(model_path="/nonexistent/model.pt", seq_len=5)

    for i in range(6):
        moved = _track("t1", (10.0 + i * 2, 10.0, 60.0 + i * 2, 60.0))
        results = predictor.predict_batch([moved], seq_len=5)

    assert len(results) == 1
    assert results[0].track_id == "t1"
    assert len(results[0].predicted_positions) == HORIZON


@pytest.mark.unit
def test_predict_batch_varying_history_lengths() -> None:
    """Only tracks meeting seq_len threshold should be predicted."""
    pytest.importorskip("torch")
    predictor = TrajectoryPredictor(model_path="/nonexistent/model.pt", seq_len=5)

    for i in range(6):
        predictor.predict_batch([_track("long", (10.0 + i, 10.0, 60.0 + i, 60.0))], seq_len=5)
    predictor.predict_batch([_track("short", (100.0, 100.0, 150.0, 150.0), age=1)], seq_len=5)

    results = predictor.predict_batch(
        [
            _track("long", (20.0, 10.0, 70.0, 60.0)),
            _track("short", (100.0, 100.0, 150.0, 150.0), age=2),
        ],
        seq_len=5,
    )
    assert len(results) == 1
    assert results[0].track_id == "long"
