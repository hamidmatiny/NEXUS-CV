"""Unit tests for the shared drift workflow and MLOps scheduler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlops.drift_workflow import (
    build_detection_dataframe,
    dump_operational_parquet,
    run_drift_check,
)
from serving.mlops_scheduler import MLOpsScheduler, reset_mlops_scheduler


@pytest.fixture
def reference_parquet(tmp_path: Path) -> Path:
    """Create reference detection parquet."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "confidence": rng.uniform(0.5, 0.95, 200),
            "bbox_area": rng.uniform(1000, 50000, 200),
            "class_id": rng.integers(0, 5, 200),
        }
    )
    path = tmp_path / "reference.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def drifted_current_df() -> pd.DataFrame:
    """Create clearly drifted current data."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "confidence": rng.uniform(0.05, 0.2, 200),
            "bbox_area": rng.uniform(90000, 120000, 200),
            "class_id": rng.integers(10, 15, 200),
        }
    )


@pytest.mark.unit
def test_dump_operational_parquet(tmp_path: Path) -> None:
    """Operational window is written to timestamped parquet."""
    df = build_detection_dataframe([0.9], [100.0], [2])
    path = dump_operational_parquet(df, tmp_path / "current")
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 1


@pytest.mark.unit
def test_run_drift_check_exit_code(
    reference_parquet: Path,
    drifted_current_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Drifted data returns exit code 1 matching CLI behaviour."""
    pytest.importorskip("evidently")
    result = run_drift_check(
        reference_path=reference_parquet,
        current_data=drifted_current_df,
        reports_dir=tmp_path / "reports",
        dataset_drift_threshold=0.3,
    )
    assert result.exit_code == 1
    assert result.dataset_drift
    assert result.report.report_path is not None
    assert result.report.report_path.exists()


@pytest.mark.unit
def test_scheduler_disabled_by_default() -> None:
    """Scheduler is inactive when MLOPS_RETRAINING_ENABLED is false."""
    reset_mlops_scheduler()
    scheduler = MLOpsScheduler()
    assert not scheduler.enabled
    assert scheduler.record_frame(MagicMock()) is None


@pytest.mark.unit
def test_scheduler_dumps_and_evaluates_on_threshold(
    reference_parquet: Path,
    drifted_current_df: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler dumps parquet and triggers orchestrator on drift."""
    pytest.importorskip("evidently")
    reset_mlops_scheduler()

    from config.settings import Settings

    settings = Settings(
        MLOPS_RETRAINING_ENABLED=True,
        MLOPS_REFERENCE_DATA_PATH=reference_parquet,
        MLOPS_REPORTS_DIR=tmp_path / "reports",
        MLOPS_TEMP_DATA_DIR=tmp_path / "current",
        MLOPS_DATASET_DRIFT_THRESHOLD=0.3,
        MLOPS_MIN_HOURS_BETWEEN_RETRAINING=0,
        MLFLOW_TRACKING_URI="http://localhost:5001",
    )
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("serving.mlops_scheduler.get_settings", lambda: settings)

    mock_orchestrator = MagicMock()
    mock_orchestrator.evaluate_from_drift_check.return_value = MagicMock(
        should_retrain=True,
        dataset_drift=True,
        drift_report_path=tmp_path / "reports" / "drift_test.html",
        mlflow_run_id="run-test",
    )

    scheduler = MLOpsScheduler(orchestrator=mock_orchestrator)
    scheduler._frame_count = 100_000 - 1

    mock_result = MagicMock()
    mock_result.detections = [
        MagicMock(confidence=0.1, bbox_xyxy=(0, 0, 10, 10), class_id=12),
    ] * 200
    mock_result.tracks = []

    decision = scheduler.record_frame(mock_result)
    assert decision is not None
    mock_orchestrator.evaluate_from_drift_check.assert_called_once()
    assert any(p.suffix == ".parquet" for p in tmp_path.glob("current/*.parquet"))


@pytest.mark.unit
def test_scheduler_respects_cooldown_via_orchestrator(
    reference_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler delegates cooldown enforcement to orchestrator."""
    reset_mlops_scheduler()
    from mlops.drift_monitor import DriftReport
    from mlops.drift_workflow import DriftCheckResult
    from mlops.retraining_orchestrator import RetrainingConfig, RetrainingDecision, RetrainingOrchestrator

    mock_monitor = MagicMock()
    orchestrator = RetrainingOrchestrator(
        drift_monitor=mock_monitor,
        experiment_tracker=MagicMock(),
        model_registry=MagicMock(),
        config=RetrainingConfig(min_hours_between_retraining=6),
    )
    orchestrator._last_retrain_at = datetime.now(tz=UTC) - timedelta(hours=1)

    drift_check = DriftCheckResult(
        report=DriftReport(
            n_drifted_features=2,
            share_drifted=0.67,
            dataset_drift=True,
            feature_reports={},
        ),
        exit_code=1,
    )
    with patch.object(orchestrator, "trigger_retraining") as mock_trigger:
        decision = orchestrator.evaluate_from_drift_check(
            drift_check=drift_check,
            embeddings=np.zeros((1, 8)),
            velocities=np.zeros(1),
        )
    assert not decision.should_retrain
    mock_trigger.assert_not_called()
