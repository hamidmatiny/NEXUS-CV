"""ViT-based scene classification with caching and graceful degradation."""

from __future__ import annotations

import hashlib

import numpy as np
import structlog
from numpy.typing import NDArray

from intelligence.data_types import ScenePrediction
from intelligence.model_utils import select_device

logger = structlog.get_logger(__name__)

SCENE_CLASSES: list[str] = ["highway", "intersection", "parking_lot", "urban_street", "tunnel"]
VIT_MODEL_ID = "google/vit-base-patch16-224"
CACHE_WINDOW = 30

# Heuristic mapping from common ImageNet-style labels to NEXUS scene classes.
_LABEL_TO_SCENE: dict[str, str] = {
    "street": "urban_street",
    "road": "highway",
    "highway": "highway",
    "bridge": "highway",
    "parking": "parking_lot",
    "lot": "parking_lot",
    "tunnel": "tunnel",
    "crosswalk": "intersection",
    "traffic": "intersection",
    "city": "urban_street",
    "downtown": "urban_street",
}


def _compute_frame_hash(frame: NDArray[np.uint8]) -> int:
    """Compute a hash bucket for 30-frame caching windows.

    Args:
        frame: BGR image array.

    Returns:
        Integer hash key for cache lookup.
    """
    small = frame[::8, ::8, 0].tobytes()
    digest = hashlib.md5(small, usedforsecurity=False).hexdigest()
    raw = int(digest[:8], 16)
    return raw // CACHE_WINDOW


def _map_label_to_scene(label: str) -> str:
    """Map a ViT label string to a NEXUS scene class.

    Args:
        label: Raw classifier label.

    Returns:
        Mapped scene class name.
    """
    lower = label.lower()
    for key, scene in _LABEL_TO_SCENE.items():
        if key in lower:
            return scene
    return SCENE_CLASSES[hash(label) % len(SCENE_CLASSES)]


class SceneClassifier:
    """Classifies driving scenes using a ViT backbone with 30-frame caching."""

    def __init__(self) -> None:
        """Initialize the scene classifier, loading ViT if available."""
        self._device = select_device()
        self._pipeline: object | None = None
        self._available = False
        self._prediction_cache: dict[int, ScenePrediction] = {}
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the Hugging Face ViT pipeline."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-classification",
                model=VIT_MODEL_ID,
                device=-1 if self._device == "cpu" else 0,
            )
            self._available = True
            logger.info("scene_classifier_loaded", model=VIT_MODEL_ID, device=self._device)
        except Exception as exc:
            self._available = False
            logger.warning("scene_classifier_degraded", error=str(exc), fallback="unknown")

    def _mock_prediction(self) -> ScenePrediction:
        """Return a degraded prediction when the model is unavailable.

        Returns:
            ScenePrediction with class ``unknown``.
        """
        return ScenePrediction(
            scene_class="unknown",
            confidence=0.0,
            top3=[("unknown", 0.0)],
        )

    def _run_vit(self, frame: NDArray[np.uint8]) -> ScenePrediction:
        """Run ViT inference on a frame.

        Args:
            frame: BGR image array.

        Returns:
            ScenePrediction from ViT logits.
        """
        if self._pipeline is None:
            return self._mock_prediction()

        rgb = frame[:, :, ::-1]
        results: list[dict[str, object]] = self._pipeline(rgb, top_k=5)  # type: ignore[operator]

        scene_scores: dict[str, float] = {cls: 0.0 for cls in SCENE_CLASSES}
        for item in results:
            label = str(item.get("label", ""))
            score = float(str(item.get("score", 0.0)))
            scene = _map_label_to_scene(label)
            scene_scores[scene] = max(scene_scores[scene], score)

        ranked = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [(name, score) for name, score in ranked[:3]]
        best_class, best_score = ranked[0]
        return ScenePrediction(scene_class=best_class, confidence=best_score, top3=top3)

    def classify(self, frame: NDArray[np.uint8]) -> ScenePrediction:
        """Classify the scene in a video frame.

        Results are cached per 30-frame hash window.

        Args:
            frame: BGR image as numpy array.

        Returns:
            ScenePrediction with top class and top-3 scores.
        """
        if not self._available:
            return self._mock_prediction()

        frame_hash = _compute_frame_hash(frame)
        cached = self._prediction_cache.get(frame_hash)
        if cached is not None:
            return cached

        result = self._run_vit(frame)
        self._prediction_cache[frame_hash] = result
        while len(self._prediction_cache) > 64:
            self._prediction_cache.pop(next(iter(self._prediction_cache)))
        return result
