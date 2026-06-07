"""Unit tests for RetrainingOrchestrator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlops.drift_monitor import DriftReport, KSDriftResult
from mlops.drift_workflow import DriftCheckResult
from mlops.retraining_orchestrator import RetrainingConfig, RetrainingOrchestrator

DETECTION_DF = pd.DataFrame({"confidence": [0.5], "bbox_area": [100.0], "class_id": [0]})


@pytest.fixture
def mock_drift_monitor() -> MagicMock:
    """Provide a mocked DriftMonitor."""
    monitor = MagicMock()
    monitor.run_detection_drift_report.return_value = DriftReport(
        n_drifted_features=2,
        share_drifted=0.67,
        dataset_drift=True,
        feature_reports={"confidence": {"drifted": True}},
    )
    monitor.run_embedding_drift.return_value = 0.05
    monitor.check_trajectory_drift.return_value = KSDriftResult(
        statistic=0.1,
        p_value=0.5,
        drift_detected=False,
    )
    return monitor


@pytest.fixture
def orchestrator(mock_drift_monitor: MagicMock) -> RetrainingOrchestrator:
    """Build a RetrainingOrchestrator with mocked dependencies."""
    mock_tracker = MagicMock()
    mock_run = MagicMock()
    mock_run.run_id = "run-test"
    mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=None)

    return RetrainingOrchestrator(
        drift_monitor=mock_drift_monitor,
        experiment_tracker=mock_tracker,
        model_registry=MagicMock(),
        config=RetrainingConfig(
            dataset_drift_threshold=0.3,
            embedding_drift_threshold=0.15,
            min_hours_between_retraining=6,
        ),
        reference_embeddings=np.ones((10, 8)),
        reference_velocities=np.zeros(50),
    )


def _drift_check_result() -> DriftCheckResult:
    """Build a drift check result indicating dataset drift."""
    from mlops.drift_workflow import DriftCheckResult

    return DriftCheckResult(
        report=DriftReport(
            n_drifted_features=2,
            share_drifted=0.67,
            dataset_drift=True,
            feature_reports={"confidence": {"drifted": True}},
            report_path=Path("reports/drift_test.html"),
        ),
        exit_code=1,
        current_data_path=Path("data/mlops/current/current_test.parquet"),
    )


@pytest.mark.unit
def test_evaluate_and_trigger_when_drift_detected(orchestrator: RetrainingOrchestrator) -> None:
    """Drift above threshold triggers retraining."""
    with patch.object(orchestrator, "trigger_retraining") as mock_trigger:
        decision = orchestrator.evaluate_and_trigger(
            detection_data=DETECTION_DF,
            embeddings=np.ones((5, 8)),
            velocities=np.array([1.0, 2.0]),
            drift_check=_drift_check_result(),
        )

    assert decision.should_retrain
    assert decision.triggered_at is not None
    assert any("dataset drift" in r for r in decision.reasons)
    mock_trigger.assert_called_once()


@pytest.mark.unit
def test_evaluate_respects_cooldown(orchestrator: RetrainingOrchestrator) -> None:
    """Cooldown blocks retraining even when drift is detected."""
    orchestrator._last_retrain_at = datetime.now(tz=UTC) - timedelta(hours=1)

    with patch.object(orchestrator, "trigger_retraining") as mock_trigger:
        decision = orchestrator.evaluate_and_trigger(
            detection_data=DETECTION_DF,
            embeddings=np.ones((5, 8)),
            velocities=np.array([1.0]),
            drift_check=_drift_check_result(),
        )

    assert not decision.should_retrain
    assert any("cooldown" in r for r in decision.reasons)
    mock_trigger.assert_not_called()


@pytest.mark.unit
def test_trigger_retraining_local_script(tmp_path: Path) -> None:
    """Without webhook, trigger_retraining invokes local training script."""
    from mlops.retraining_orchestrator import RetrainingDecision, RetrainingOrchestrator

    script = tmp_path / "train_trajectory_lstm.py"
    script.write_text("# stub")

    mock_tracker = MagicMock()
    mock_run = MagicMock()
    mock_run.run_id = "run-local"
    mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=None)

    orchestrator = RetrainingOrchestrator(
        drift_monitor=MagicMock(),
        experiment_tracker=mock_tracker,
        model_registry=MagicMock(),
        config=RetrainingConfig(),
    )
    decision = RetrainingDecision(
        should_retrain=True,
        reasons=["dataset drift"],
        drift_scores={"share_drifted": 0.5},
        triggered_at=datetime.now(tz=UTC),
        drift_report_path=tmp_path / "drift.html",
    )
    decision.drift_report_path.write_text("<html></html>")

    with (
        patch("mlops.retraining_orchestrator.TRAIN_SCRIPT", script),
        patch("mlops.retraining_orchestrator.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        orchestrator.trigger_retraining(decision)

    mock_run.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trigger_retraining_webhook(orchestrator: RetrainingOrchestrator) -> None:
    """Webhook URL triggers async POST with retry."""
    from unittest.mock import AsyncMock

    orchestrator._config.webhook_url = "http://example.com/retrain"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await orchestrator._post_webhook({"test": True})

    mock_client.post.assert_called_once()
