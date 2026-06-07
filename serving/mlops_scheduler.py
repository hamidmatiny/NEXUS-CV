"""Scheduled MLOps drift evaluation and automated retraining from the serving layer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from config.settings import get_settings
from mlops.drift_workflow import (
    DETECTION_FEATURE_COLUMNS,
    build_detection_dataframe,
    dump_operational_parquet,
    run_drift_check,
)

if TYPE_CHECKING:
    from mlops.retraining_orchestrator import RetrainingDecision, RetrainingOrchestrator

logger = structlog.get_logger(__name__)

FRAME_CHECK_INTERVAL = 100_000
TIME_CHECK_INTERVAL_S = 3600.0


@dataclass
class ServingObservation:
    """Rolling observation buffer for drift evaluation."""

    confidences: list[float] = field(default_factory=list)
    bbox_areas: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


class MLOpsScheduler:
    """Triggers drift evaluation every 100k frames or 1 hour (whichever first)."""

    def __init__(self, orchestrator: RetrainingOrchestrator | None = None) -> None:
        """Initialize the scheduler.

        Args:
            orchestrator: Optional RetrainingOrchestrator (lazy-loaded when None).
        """
        self._orchestrator = orchestrator
        self._frame_count = 0
        self._last_check_monotonic = time.monotonic()
        self._observations = ServingObservation()
        self._settings = get_settings()
        self._enabled = self._settings.MLOPS_RETRAINING_ENABLED

        if self._enabled:
            logger.info(
                "mlops_scheduler_enabled",
                reference_data=str(self._settings.MLOPS_REFERENCE_DATA_PATH),
                reports_dir=str(self._settings.MLOPS_REPORTS_DIR),
                mlflow_uri=self._settings.MLFLOW_TRACKING_URI,
            )
        else:
            logger.debug("mlops_scheduler_disabled")

    @property
    def enabled(self) -> bool:
        """Return whether automated retraining is enabled."""
        return self._enabled

    def record_frame(self, pipeline_result: Any) -> RetrainingDecision | None:
        """Record a pipeline result and evaluate drift if thresholds are met.

        Args:
            pipeline_result: Completed PipelineResult from inference.

        Returns:
            RetrainingDecision if evaluation ran, else None.
        """
        if not self._enabled:
            return None

        self._frame_count += 1
        self._accumulate(pipeline_result)

        if not self._should_run_evaluation():
            return None

        return self._run_evaluation_window(trigger="frame_or_time_threshold")

    def check_time_based(self) -> RetrainingDecision | None:
        """Evaluate drift when the hourly interval elapses (background loop).

        Returns:
            RetrainingDecision if evaluation ran, else None.
        """
        if not self._enabled:
            return None

        elapsed = time.monotonic() - self._last_check_monotonic
        if elapsed < TIME_CHECK_INTERVAL_S:
            return None

        if not self._observations.confidences:
            self._last_check_monotonic = time.monotonic()
            logger.debug("mlops_hourly_skip_no_observations")
            return None

        return self._run_evaluation_window(trigger="hourly_background")

    def _should_run_evaluation(self) -> bool:
        """Return True when frame count or elapsed time thresholds are met."""
        elapsed = time.monotonic() - self._last_check_monotonic
        return self._frame_count >= FRAME_CHECK_INTERVAL or elapsed >= TIME_CHECK_INTERVAL_S

    def _accumulate(self, result: Any) -> None:
        """Append detection and track stats from a pipeline result.

        Args:
            result: PipelineResult with detections and tracks.
        """
        for det in result.detections:
            self._observations.confidences.append(float(det.confidence))
            x1, y1, x2, y2 = det.bbox_xyxy
            self._observations.bbox_areas.append(float((x2 - x1) * (y2 - y1)))
            self._observations.class_ids.append(int(det.class_id))

        for track in result.tracks:
            vx, vy = track.velocity_2d
            self._observations.velocities.append(float(np.hypot(vx, vy)))

    def _run_evaluation_window(self, trigger: str) -> RetrainingDecision | None:
        """Dump operational window, run drift check, and optionally trigger retraining.

        Args:
            trigger: Human-readable trigger reason for logging.

        Returns:
            RetrainingDecision or None if evaluation could not complete.
        """
        self._frame_count = 0
        self._last_check_monotonic = time.monotonic()

        obs = self._observations
        if not obs.confidences:
            logger.debug("mlops_skip_empty_observations", trigger=trigger)
            return None

        reference_path = Path(self._settings.MLOPS_REFERENCE_DATA_PATH)
        if not reference_path.exists():
            logger.warning("mlops_reference_data_missing", path=str(reference_path))
            return None

        detection_data = build_detection_dataframe(
            obs.confidences,
            obs.bbox_areas,
            obs.class_ids,
            obs.velocities or None,
        )
        current_path = dump_operational_parquet(detection_data, self._settings.MLOPS_TEMP_DATA_DIR)

        drift_check = run_drift_check(
            reference_path=reference_path,
            current_data=detection_data,
            reports_dir=self._settings.MLOPS_REPORTS_DIR,
            dataset_drift_threshold=self._settings.MLOPS_DATASET_DRIFT_THRESHOLD,
            feature_columns=DETECTION_FEATURE_COLUMNS,
            current_data_path=current_path,
        )

        if obs.embeddings:
            embeddings = np.array(obs.embeddings, dtype=np.float32)
        else:
            embeddings = np.zeros((1, 8), dtype=np.float32)
        velocities = (
            np.array(obs.velocities, dtype=np.float32) if obs.velocities else np.zeros(1)
        )

        orchestrator = self._orchestrator or _build_orchestrator()
        if orchestrator is None:
            logger.warning("mlops_orchestrator_unavailable")
            self._observations = ServingObservation()
            return None

        decision = orchestrator.evaluate_from_drift_check(
            drift_check=drift_check,
            embeddings=embeddings,
            velocities=velocities,
        )
        self._observations = ServingObservation()

        logger.info(
            "mlops_scheduled_evaluation_complete",
            trigger=trigger,
            should_retrain=decision.should_retrain,
            dataset_drift=decision.dataset_drift,
            drift_exit_code=drift_check.exit_code,
            drift_report_path=str(decision.drift_report_path)
            if decision.drift_report_path
            else None,
            mlflow_run_id=decision.mlflow_run_id,
        )
        return decision


_scheduler: MLOpsScheduler | None = None


def get_mlops_scheduler() -> MLOpsScheduler:
    """Return the process-wide MLOps scheduler singleton.

    Returns:
        MLOpsScheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = MLOpsScheduler()
    return _scheduler


def reset_mlops_scheduler() -> None:
    """Reset the scheduler singleton (for tests)."""
    global _scheduler
    _scheduler = None


def _build_orchestrator() -> RetrainingOrchestrator | None:
    """Build a RetrainingOrchestrator from settings when reference data exists.

    Returns:
        Configured orchestrator or None.
    """
    settings = get_settings()
    ref_path = Path(settings.MLOPS_REFERENCE_DATA_PATH)
    if not ref_path.exists():
        return None

    try:
        from mlops.drift_monitor import DriftMonitor
        from mlops.experiment_tracker import NexusExperimentTracker
        from mlops.model_registry import ModelRegistry
        from mlops.retraining_orchestrator import RetrainingConfig, RetrainingOrchestrator

        monitor = DriftMonitor(
            reference_dataset_path=ref_path,
            feature_columns=DETECTION_FEATURE_COLUMNS,
            reports_dir=settings.MLOPS_REPORTS_DIR,
            dataset_drift_threshold=settings.MLOPS_DATASET_DRIFT_THRESHOLD,
        )
        tracker = NexusExperimentTracker(tracking_uri=settings.MLFLOW_TRACKING_URI)
        registry = ModelRegistry(tracking_uri=settings.MLFLOW_TRACKING_URI)
        config = RetrainingConfig(
            dataset_drift_threshold=settings.MLOPS_DATASET_DRIFT_THRESHOLD,
            embedding_drift_threshold=settings.MLOPS_EMBEDDING_DRIFT_THRESHOLD,
            min_hours_between_retraining=settings.MLOPS_MIN_HOURS_BETWEEN_RETRAINING,
            webhook_url=settings.MLOPS_RETRAINING_WEBHOOK_URL,
        )
        ref_df = monitor.reference_data
        ref_velocities = ref_df["velocity"].to_numpy() if "velocity" in ref_df.columns else None
        return RetrainingOrchestrator(
            drift_monitor=monitor,
            experiment_tracker=tracker,
            model_registry=registry,
            config=config,
            reference_velocities=ref_velocities,
        )
    except Exception as exc:
        logger.warning("mlops_orchestrator_init_failed", error=str(exc))
        return None
