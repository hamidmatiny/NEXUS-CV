"""Unit tests for SceneClassifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intelligence.data_types import ScenePrediction
from intelligence.scene_classifier import SceneClassifier, _compute_frame_hash


@pytest.mark.unit
def test_graceful_degradation_returns_unknown(synthetic_frame: np.ndarray) -> None:
    """When ViT fails to load, classify should return unknown scene."""
    with patch.object(
        SceneClassifier, "_try_load_model", lambda self: setattr(self, "_available", False)
    ):
        classifier = SceneClassifier()
        result = classifier.classify(synthetic_frame)
    assert isinstance(result, ScenePrediction)
    assert result.scene_class == "unknown"
    assert result.confidence == 0.0


@pytest.mark.unit
def test_cache_returns_same_result(synthetic_frame: np.ndarray) -> None:
    """Identical frame hash bucket should return cached prediction."""
    mock_prediction = ScenePrediction(
        scene_class="highway",
        confidence=0.9,
        top3=[("highway", 0.9), ("urban_street", 0.05), ("tunnel", 0.05)],
    )
    with patch.object(
        SceneClassifier, "_try_load_model", lambda self: setattr(self, "_available", True)
    ):
        classifier = SceneClassifier()
        with patch.object(classifier, "_run_vit", return_value=mock_prediction):
            first = classifier.classify(synthetic_frame)
            second = classifier.classify(synthetic_frame)
    assert first.scene_class == second.scene_class
    assert first.confidence == second.confidence


@pytest.mark.unit
def test_frame_hash_bucket_stable() -> None:
    """Frame hash should be stable for identical frames."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _compute_frame_hash(frame) == _compute_frame_hash(frame.copy())


@pytest.mark.unit
def test_run_vit_mocked_pipeline(synthetic_frame: np.ndarray) -> None:
    """Mocked ViT pipeline should produce a valid ScenePrediction."""
    mock_pipeline = MagicMock(return_value=[{"label": "highway road", "score": 0.85}])
    with patch.object(SceneClassifier, "_try_load_model", lambda self: None):
        classifier = SceneClassifier()
        classifier._available = True
        classifier._pipeline = mock_pipeline
        result = classifier._run_vit(synthetic_frame)
    assert result.scene_class in {
        "highway",
        "intersection",
        "parking_lot",
        "urban_street",
        "tunnel",
    }
    assert result.confidence > 0.0
