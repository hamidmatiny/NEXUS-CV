"""Unit tests for AnomalyScorer."""

from __future__ import annotations

from collections import Counter

import pytest

from fusion.data_types import Track
from intelligence.anomaly_scorer import ANOMALY_THRESHOLD, AnomalyScorer, _SceneStats
from intelligence.data_types import ScenePrediction


def _track(
    track_id: str,
    bbox: tuple[float, float, float, float],
    velocity: tuple[float, float] = (1.0, 0.0),
    state: str = "confirmed",
    age: int = 10,
    class_name: str = "car",
) -> Track:
    """Build a test Track."""
    return Track(
        track_id=track_id,
        state=state,  # type: ignore[arg-type]
        age_frames=age,
        modalities_seen={"camera"},
        last_bbox_2d=bbox,
        last_bbox_3d=None,
        velocity_2d=velocity,
        class_votes=Counter({class_name: age}),
    )


def _scene(scene_class: str = "highway") -> ScenePrediction:
    """Build a test ScenePrediction."""
    return ScenePrediction(scene_class=scene_class, confidence=0.9, top3=[(scene_class, 0.9)])


@pytest.fixture
def scorer() -> AnomalyScorer:
    """Provide a fresh AnomalyScorer."""
    return AnomalyScorer()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factor_name", "setup_fn"),
    [
        (
            "velocity_anomaly",
            lambda s: _velocity_anomaly_setup(s),
        ),
        (
            "wrong_class_speed",
            lambda s: _wrong_class_speed_setup(s),
        ),
        (
            "near_miss",
            lambda s: _near_miss_setup(s),
        ),
        (
            "track_resurrection",
            lambda s: _resurrection_setup(s),
        ),
    ],
)
def test_contributing_factor_isolated(
    scorer: AnomalyScorer,
    factor_name: str,
    setup_fn: object,
) -> None:
    """Each contributing factor should fire independently."""
    track, scene, all_tracks = setup_fn(scorer)  # type: ignore[operator]
    result = scorer.score(track, scene, all_tracks)
    assert any(
        factor_name in f for f in result.contributing_factors
    ), f"Expected factor {factor_name}, got {result.contributing_factors}"
    assert result.score > 0.0


def _velocity_anomaly_setup(scorer: AnomalyScorer) -> tuple[Track, ScenePrediction, list[Track]]:
    """Build scenario for velocity z-score anomaly."""
    scene = _scene("highway")
    normal = [_track(f"n{i}", (10.0, 10.0, 50.0, 50.0), velocity=(1.0, 0.0)) for i in range(20)]
    for t in normal:
        scorer.score(t, scene, normal)
    fast = _track("fast", (10.0, 10.0, 50.0, 50.0), velocity=(50.0, 0.0))
    return fast, scene, normal + [fast]


def _wrong_class_speed_setup(scorer: AnomalyScorer) -> tuple[Track, ScenePrediction, list[Track]]:
    """Build scenario for person at highway speed."""
    scene = _scene("highway")
    person = _track("runner", (10.0, 10.0, 50.0, 50.0), velocity=(30.0, 0.0), class_name="person")
    return person, scene, [person]


def _near_miss_setup(scorer: AnomalyScorer) -> tuple[Track, ScenePrediction, list[Track]]:
    """Build scenario for near-miss IoU between different classes."""
    scene = _scene("intersection")
    car = _track("car1", (100.0, 100.0, 200.0, 200.0), class_name="car")
    person = _track("person1", (110.0, 110.0, 190.0, 190.0), class_name="person")
    return car, scene, [car, person]


def _resurrection_setup(scorer: AnomalyScorer) -> tuple[Track, ScenePrediction, list[Track]]:
    """Build scenario for track resurrection within 5 frames."""
    scene = _scene("urban_street")
    scorer.register_dead_track("dead-001")
    reborn = _track("reborn-002", (50.0, 50.0, 100.0, 100.0), age=1)
    return reborn, scene, [reborn]


@pytest.mark.unit
def test_anomaly_threshold(scorer: AnomalyScorer) -> None:
    """Score >= 0.65 should mark track as anomalous."""
    scene = _scene("highway")
    person = _track(
        "fast_person", (10.0, 10.0, 50.0, 50.0), velocity=(30.0, 0.0), class_name="person"
    )
    car = _track("car1", (100.0, 100.0, 200.0, 200.0), class_name="car")
    person2 = _track("person1", (110.0, 110.0, 190.0, 190.0), class_name="person")
    all_tracks = [person, car, person2]
    result = scorer.score(person, scene, all_tracks)
    if result.score >= ANOMALY_THRESHOLD:
        assert result.is_anomalous is True
    else:
        assert result.is_anomalous is False


@pytest.mark.unit
def test_scene_stats_velocity_zscore() -> None:
    """Velocity z-score should be zero with insufficient data."""
    stats = _SceneStats()
    assert stats.velocity_zscore(5.0) == 0.0
