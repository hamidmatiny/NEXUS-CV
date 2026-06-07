"""Unit tests for NexusExperimentTracker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.unit
def test_start_run_sets_tags() -> None:
    """start_run binds git/python/ray tags and returns a context manager."""
    mock_run = MagicMock()
    mock_run.info.run_id = "run-123"

    with (
        patch("mlflow.start_run", return_value=mock_run) as mock_start,
        patch("mlflow.set_experiment"),
        patch("mlflow.set_tags") as mock_tags,
        patch("mlflow.end_run") as mock_end,
    ):
        from mlops.experiment_tracker import NexusExperimentTracker

        tracker = NexusExperimentTracker(experiment_name="test-exp")
        with tracker.start_run("yolo", "run-a") as run:
            assert run.run_id == "run-123"

    mock_start.assert_called_once_with(run_name="run-a")
    mock_tags.assert_called_once()
    tag_keys = set(mock_tags.call_args[0][0].keys())
    assert {"model_name", "git_commit", "python_version", "ray_version"} <= tag_keys
    mock_end.assert_called_once()


@pytest.mark.unit
def test_log_detection_run_logs_metrics_and_artifacts(tmp_path: Path) -> None:
    """log_detection_run logs required detection metrics."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")

    with (
        patch("mlflow.log_metrics") as mock_metrics,
        patch("mlflow.log_artifact"),
        patch("mlops.experiment_tracker.plt") as mock_plt,
    ):
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())
        from mlops.experiment_tracker import NexusExperimentTracker

        tracker = NexusExperimentTracker()
        tracker.log_detection_run(
            model_path=model_path,
            val_metrics={
                "mAP50": 0.72,
                "mAP50-95": 0.55,
                "precision": 0.81,
                "recall": 0.76,
                "inference_ms_p99": 28.0,
            },
            confusion_matrix=np.eye(3),
            sample_frames=[np.zeros((32, 32, 3), dtype=np.uint8)],
        )

    logged = mock_metrics.call_args[0][0]
    assert logged["mAP50"] == 0.72
    assert logged["inference_ms_p99"] == 28.0


@pytest.mark.unit
def test_log_lstm_run_logs_ade_fde() -> None:
    """log_lstm_run logs ADE and FDE trajectory metrics."""
    with (
        patch("mlflow.log_metrics") as mock_metrics,
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("mlops.experiment_tracker.plt") as mock_plt,
    ):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        from mlops.experiment_tracker import NexusExperimentTracker

        tracker = NexusExperimentTracker()
        tracker.log_lstm_run(
            model_path=Path("missing.pt"),
            train_loss_curve=[1.0, 0.5],
            val_loss_curve=[1.2, 0.7],
            ade_m=1.1,
            fde_m=2.0,
        )

    logged = mock_metrics.call_args[0][0]
    assert logged["ade_m"] == 1.1
    assert logged["fde_m"] == 2.0


@pytest.mark.unit
def test_log_system_benchmark() -> None:
    """log_system_benchmark logs serving latency percentiles."""
    with patch("mlflow.log_metrics") as mock_metrics:
        from mlops.experiment_tracker import NexusExperimentTracker

        tracker = NexusExperimentTracker()
        tracker.log_system_benchmark(
            serving_percentiles={"p50": 10.0, "p95": 25.0, "p99": 40.0},
            sla_breach_rate=0.02,
            active_tracks_mean=12.5,
        )

    logged = mock_metrics.call_args[0][0]
    assert logged["serving_p99_ms"] == 40.0
    assert logged["sla_breach_rate"] == 0.02
