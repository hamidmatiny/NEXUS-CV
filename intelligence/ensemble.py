"""Orchestrates scene classification, trajectory prediction, and anomaly scoring."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import structlog
from numpy.typing import NDArray

from fusion.data_types import Track
from intelligence.anomaly_scorer import AnomalyScorer
from intelligence.data_types import AnomalyScore, IntelligenceOutput
from intelligence.scene_classifier import SceneClassifier
from intelligence.trajectory_lstm import TrajectoryPredictor

logger = structlog.get_logger(__name__)


class IntelligenceEnsemble:
    """Stacked intelligence layer operating on fused tracks."""

    def __init__(
        self,
        scene_classifier: SceneClassifier | None = None,
        trajectory_predictor: TrajectoryPredictor | None = None,
        anomaly_scorer: AnomalyScorer | None = None,
    ) -> None:
        """Initialize intelligence sub-components.

        Args:
            scene_classifier: Scene classifier instance.
            trajectory_predictor: Trajectory predictor instance.
            anomaly_scorer: Anomaly scorer instance.
        """
        self._scene = scene_classifier or SceneClassifier()
        self._trajectory = trajectory_predictor or TrajectoryPredictor()
        self._anomaly = anomaly_scorer or AnomalyScorer()

    def run(
        self,
        frame: NDArray[np.uint8],
        tracks: list[Track],
        frame_id: int = 0,
        camera_id: str = "cam_00",
    ) -> IntelligenceOutput:
        """Run the full intelligence pipeline synchronously.

        Args:
            frame: BGR video frame.
            tracks: Fused tracks from the fusion layer.
            frame_id: Frame identifier.
            camera_id: Camera identifier.

        Returns:
            IntelligenceOutput aggregating all predictions.
        """
        start = time.perf_counter()
        confirmed = [t for t in tracks if t.state == "confirmed"]

        scene = self._scene.classify(frame)
        trajectories = self._trajectory.predict_batch(confirmed)

        anomalies: list[AnomalyScore] = []
        for track in confirmed:
            anomalies.append(self._anomaly.score(track, scene, confirmed))
        self._anomaly.advance_frame()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = IntelligenceOutput(
            frame_id=frame_id,
            camera_id=camera_id,
            scene=scene,
            trajectories=trajectories,
            anomalies=anomalies,
            inference_total_ms=round(elapsed_ms, 2),
        )
        logger.info(
            "intelligence_inference_complete",
            frame_id=frame_id,
            camera_id=camera_id,
            inference_total_ms=output.inference_total_ms,
            num_trajectories=len(trajectories),
            num_anomalies=sum(1 for a in anomalies if a.is_anomalous),
        )
        return output

    async def run_async(
        self,
        frame: NDArray[np.uint8],
        tracks: list[Track],
        frame_id: int = 0,
        camera_id: str = "cam_00",
    ) -> IntelligenceOutput:
        """Run scene and trajectory inference in parallel via asyncio.

        Args:
            frame: BGR video frame.
            tracks: Fused tracks from the fusion layer.
            frame_id: Frame identifier.
            camera_id: Camera identifier.

        Returns:
            IntelligenceOutput aggregating all predictions.
        """
        start = time.perf_counter()
        confirmed = [t for t in tracks if t.state == "confirmed"]

        scene_task = asyncio.to_thread(self._scene.classify, frame)
        trajectory_task = asyncio.to_thread(self._trajectory.predict_batch, confirmed)
        scene, trajectories = await asyncio.gather(scene_task, trajectory_task)

        anomalies: list[AnomalyScore] = []
        for track in confirmed:
            anomalies.append(self._anomaly.score(track, scene, confirmed))
        self._anomaly.advance_frame()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = IntelligenceOutput(
            frame_id=frame_id,
            camera_id=camera_id,
            scene=scene,
            trajectories=trajectories,
            anomalies=anomalies,
            inference_total_ms=round(elapsed_ms, 2),
        )
        logger.info(
            "intelligence_inference_complete",
            frame_id=frame_id,
            camera_id=camera_id,
            inference_total_ms=output.inference_total_ms,
            async_mode=True,
        )
        return output
