"""Automated retraining orchestration based on drift signals."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from mlops.drift_workflow import DriftCheckResult

if TYPE_CHECKING:
    from mlops.drift_monitor import DriftMonitor, KSDriftResult
    from mlops.experiment_tracker import NexusExperimentTracker
    from mlops.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)

TRAIN_SCRIPT = Path("models/train_trajectory_lstm.py")


class RetrainingConfig(BaseModel):
    """Configuration for automated retraining triggers."""

    dataset_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_drift_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    min_hours_between_retraining: int = Field(default=6, ge=0)
    webhook_url: str | None = None


@dataclass
class RetrainingDecision:
    """Outcome of drift evaluation for retraining."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    triggered_at: datetime | None = None
    dataset_drift: bool = False
    drift_report_path: Path | None = None
    current_data_path: Path | None = None
    mlflow_run_id: str | None = None


class RetrainingOrchestrator:
    """Evaluates drift signals and triggers retraining when thresholds are breached."""

    def __init__(
        self,
        drift_monitor: DriftMonitor,
        experiment_tracker: NexusExperimentTracker,
        model_registry: ModelRegistry,
        config: RetrainingConfig,
        reference_embeddings: NDArray[np.floating] | None = None,
        reference_velocities: NDArray[np.floating] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            drift_monitor: DriftMonitor instance.
            experiment_tracker: NexusExperimentTracker instance.
            model_registry: ModelRegistry instance.
            config: Retraining configuration.
            reference_embeddings: Optional reference ViT embeddings.
            reference_velocities: Optional reference velocity samples.
        """
        self._drift_monitor = drift_monitor
        self._experiment_tracker = experiment_tracker
        self._model_registry = model_registry
        self._config = config
        self._reference_embeddings = reference_embeddings
        self._reference_velocities = reference_velocities
        self._last_retrain_at: datetime | None = None
        logger.info("retraining_orchestrator_initialized", config=config.model_dump())

    @property
    def last_retrain_at(self) -> datetime | None:
        """Return timestamp of last retraining trigger."""
        return self._last_retrain_at

    def evaluate_and_trigger(
        self,
        detection_data: pd.DataFrame,
        embeddings: NDArray[np.floating],
        velocities: NDArray[np.floating],
        drift_check: DriftCheckResult | None = None,
    ) -> RetrainingDecision:
        """Run drift checks and decide whether to retrain.

        Args:
            detection_data: Current detection feature DataFrame.
            embeddings: Current ViT embeddings.
            velocities: Current velocity samples.
            drift_check: Optional pre-computed drift check (from drift workflow).

        Returns:
            RetrainingDecision with drift scores and retrain flag.
        """
        if drift_check is None:
            from config.settings import get_settings
            from mlops.drift_workflow import dump_operational_parquet, run_drift_check

            settings = get_settings()
            current_path = dump_operational_parquet(detection_data, settings.MLOPS_TEMP_DATA_DIR)
            drift_check = run_drift_check(
                reference_path=Path(settings.MLOPS_REFERENCE_DATA_PATH),
                current_data=detection_data,
                reports_dir=settings.MLOPS_REPORTS_DIR,
                dataset_drift_threshold=self._config.dataset_drift_threshold,
                current_data_path=current_path,
            )

        return self.evaluate_from_drift_check(
            drift_check=drift_check,
            embeddings=embeddings,
            velocities=velocities,
        )

    def evaluate_from_drift_check(
        self,
        drift_check: DriftCheckResult,
        embeddings: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> RetrainingDecision:
        """Evaluate retraining decision from a completed drift check.

        Args:
            drift_check: DriftCheckResult from the shared drift workflow.
            embeddings: Current ViT embeddings.
            velocities: Current velocity samples.

        Returns:
            RetrainingDecision with audit metadata.
        """
        report = drift_check.report
        drift_scores: dict[str, float] = {
            "share_drifted": report.share_drifted,
            "n_drifted_features": float(report.n_drifted_features),
            "drift_exit_code": float(drift_check.exit_code),
        }

        reasons: list[str] = []
        if report.dataset_drift or drift_check.exit_code == 1:
            reasons.append(
                f"dataset drift: share_drifted={report.share_drifted:.3f}, "
                f"exit_code={drift_check.exit_code}"
            )

        ref_embeddings = self._reference_embeddings
        if ref_embeddings is not None and len(embeddings) > 0:
            embedding_score = self._drift_monitor.run_embedding_drift(ref_embeddings, embeddings)
            drift_scores["embedding_drift"] = embedding_score
            if embedding_score >= self._config.embedding_drift_threshold:
                reasons.append(
                    f"embedding drift: score={embedding_score:.3f} "
                    f">= {self._config.embedding_drift_threshold}"
                )

        ref_velocities = self._reference_velocities
        if ref_velocities is not None and len(velocities) > 0:
            ks_result = self._drift_monitor.check_trajectory_drift(ref_velocities, velocities)
            drift_scores["velocity_ks_statistic"] = ks_result.statistic
            drift_scores["velocity_ks_pvalue"] = ks_result.p_value
            if ks_result.drift_detected:
                reasons.append(f"trajectory drift: KS p={ks_result.p_value:.4f} < 0.05")

        drift_detected = report.dataset_drift or drift_check.exit_code == 1 or bool(
            [r for r in reasons if r.startswith(("embedding", "trajectory"))]
        )
        should_retrain = drift_detected and bool(reasons) and self._cooldown_elapsed()
        if reasons and not self._cooldown_elapsed():
            min_hours = self._config.min_hours_between_retraining
            reasons.append(f"cooldown active: min {min_hours}h between retrains")
            should_retrain = False

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            reasons=reasons,
            drift_scores=drift_scores,
            triggered_at=datetime.now(tz=UTC) if should_retrain else None,
            dataset_drift=report.dataset_drift,
            drift_report_path=report.report_path,
            current_data_path=drift_check.current_data_path,
        )

        logger.info(
            "retraining_evaluation_complete",
            should_retrain=should_retrain,
            dataset_drift=report.dataset_drift,
            drift_exit_code=drift_check.exit_code,
            drift_report_path=str(report.report_path) if report.report_path else None,
            reasons=reasons,
            drift_scores=drift_scores,
        )

        if should_retrain:
            self.trigger_retraining(decision)

        return decision

    def trigger_retraining(self, decision: RetrainingDecision) -> None:
        """Trigger retraining via webhook or local training script.

        Logs MLflow run ID, drift report path, and operational window path for audit.

        Args:
            decision: RetrainingDecision that passed drift and cooldown checks.
        """
        self._last_retrain_at = datetime.now(tz=UTC)
        run_name = f"auto-retrain-{self._last_retrain_at.strftime('%Y%m%d_%H%M%S')}"

        with self._experiment_tracker.start_run("trajectory_lstm", run_name) as run:
            decision.mlflow_run_id = run.run_id
            self._log_retraining_artifacts(decision)

            payload = {
                "should_retrain": decision.should_retrain,
                "reasons": decision.reasons,
                "drift_scores": decision.drift_scores,
                "dataset_drift": decision.dataset_drift,
                "triggered_at": (
                    decision.triggered_at.isoformat() if decision.triggered_at else None
                ),
                "mlflow_run_id": decision.mlflow_run_id,
                "drift_report_path": (
                    str(decision.drift_report_path) if decision.drift_report_path else None
                ),
                "current_data_path": (
                    str(decision.current_data_path) if decision.current_data_path else None
                ),
            }

            if self._config.webhook_url:
                asyncio.run(self._post_webhook(payload))
            else:
                self._run_local_training()

        logger.info(
            "retraining_triggered",
            mlflow_run_id=decision.mlflow_run_id,
            drift_report_path=str(decision.drift_report_path)
            if decision.drift_report_path
            else None,
            current_data_path=str(decision.current_data_path)
            if decision.current_data_path
            else None,
            reasons=decision.reasons,
        )

    def _log_retraining_artifacts(self, decision: RetrainingDecision) -> None:
        """Log drift audit artifacts to the active MLflow run.

        Args:
            decision: Retraining decision with audit paths.
        """
        import mlflow

        mlflow.log_params(
            {
                "trigger_reasons": "; ".join(decision.reasons),
                "dataset_drift": str(decision.dataset_drift),
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in decision.drift_scores.items() if isinstance(v, (int, float))}
        )
        if decision.drift_report_path and decision.drift_report_path.exists():
            mlflow.log_artifact(str(decision.drift_report_path), artifact_path="drift_reports")
        if decision.current_data_path and decision.current_data_path.exists():
            mlflow.log_artifact(str(decision.current_data_path), artifact_path="operational_windows")

    async def _post_webhook(self, payload: dict[str, Any], max_retries: int = 3) -> None:
        """POST retraining payload to webhook with retries.

        Args:
            payload: JSON payload.
            max_retries: Maximum retry attempts.
        """
        url = self._config.webhook_url
        if not url:
            return

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                logger.info("retraining_webhook_success", url=url, attempt=attempt)
                return
            except httpx.HTTPError as exc:
                logger.warning(
                    "retraining_webhook_failed",
                    url=url,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)

    def _run_local_training(self) -> None:
        """Invoke the local TrajectoryLSTM training script."""
        script = TRAIN_SCRIPT.resolve()
        if not script.exists():
            logger.error("training_script_missing", path=str(script))
            return

        cmd = [sys.executable, str(script)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
            )
            logger.info(
                "local_retraining_complete",
                returncode=result.returncode,
                stdout_tail=result.stdout[-500:] if result.stdout else "",
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "local_retraining_failed",
                returncode=exc.returncode,
                stderr=exc.stderr[-500:] if exc.stderr else "",
            )
        except subprocess.TimeoutExpired:
            logger.error("local_retraining_timeout", script=str(script))

    def _cooldown_elapsed(self) -> bool:
        """Check whether the retraining cooldown period has elapsed.

        Returns:
            True if retraining is allowed by cooldown policy.
        """
        if self._last_retrain_at is None:
            return True
        elapsed = datetime.now(tz=UTC) - self._last_retrain_at
        return elapsed >= timedelta(hours=self._config.min_hours_between_retraining)
