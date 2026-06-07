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
    ) -> RetrainingDecision:
        """Run drift checks and decide whether to retrain.

        Args:
            detection_data: Current detection feature DataFrame.
            embeddings: Current ViT embeddings.
            velocities: Current velocity samples.

        Returns:
            RetrainingDecision with drift scores and retrain flag.
        """
        drift_report = self._drift_monitor.run_detection_drift_report(detection_data)
        drift_scores: dict[str, float] = {
            "share_drifted": drift_report.share_drifted,
            "n_drifted_features": float(drift_report.n_drifted_features),
        }

        reasons: list[str] = []
        if drift_report.share_drifted >= self._config.dataset_drift_threshold:
            reasons.append(
                f"dataset drift: share_drifted={drift_report.share_drifted:.3f} "
                f">= {self._config.dataset_drift_threshold}"
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
        ks_result: KSDriftResult | None = None
        if ref_velocities is not None and len(velocities) > 0:
            ks_result = self._drift_monitor.check_trajectory_drift(ref_velocities, velocities)
            drift_scores["velocity_ks_statistic"] = ks_result.statistic
            drift_scores["velocity_ks_pvalue"] = ks_result.p_value
            if ks_result.drift_detected:
                reasons.append(
                    f"trajectory drift: KS p={ks_result.p_value:.4f} < 0.05"
                )

        should_retrain = bool(reasons) and self._cooldown_elapsed()
        if reasons and not self._cooldown_elapsed():
            min_hours = self._config.min_hours_between_retraining
            reasons.append(f"cooldown active: min {min_hours}h between retrains")
            should_retrain = False

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            reasons=reasons,
            drift_scores=drift_scores,
            triggered_at=datetime.now(tz=UTC) if should_retrain else None,
        )

        logger.info(
            "retraining_evaluation_complete",
            should_retrain=should_retrain,
            reasons=reasons,
            drift_scores=drift_scores,
        )

        if should_retrain:
            self.trigger_retraining(decision)

        return decision

    def trigger_retraining(self, decision: RetrainingDecision) -> None:
        """Trigger retraining via webhook or local training script.

        Args:
            decision: RetrainingDecision that passed drift and cooldown checks.
        """
        self._last_retrain_at = datetime.now(tz=UTC)
        payload = {
            "should_retrain": decision.should_retrain,
            "reasons": decision.reasons,
            "drift_scores": decision.drift_scores,
            "triggered_at": decision.triggered_at.isoformat() if decision.triggered_at else None,
        }

        if self._config.webhook_url:
            asyncio.run(self._post_webhook(payload))
        else:
            self._run_local_training()

        logger.info("retraining_triggered", payload=payload)

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
