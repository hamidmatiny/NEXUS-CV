"""Unit tests for DriftMonitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def reference_parquet(tmp_path: Path) -> Path:
    """Create a reference detection features parquet file."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "confidence": rng.uniform(0.5, 0.95, 200),
            "bbox_area": rng.uniform(1000, 50000, 200),
            "class_id": rng.integers(0, 5, 200),
            "velocity": rng.uniform(0, 10, 200),
        }
    )
    path = tmp_path / "reference.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def current_parquet_similar(reference_parquet: Path, tmp_path: Path) -> Path:
    """Create current data similar to reference."""
    ref = pd.read_parquet(reference_parquet)
    path = tmp_path / "current.parquet"
    ref.sample(100, random_state=1).to_parquet(path)
    return path


@pytest.fixture
def current_parquet_drifted(tmp_path: Path) -> Path:
    """Create current data with shifted distributions."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "confidence": rng.uniform(0.1, 0.4, 200),
            "bbox_area": rng.uniform(80000, 120000, 200),
            "class_id": rng.integers(10, 15, 200),
        }
    )
    path = tmp_path / "current_drifted.parquet"
    df.to_parquet(path)
    return path


@pytest.mark.unit
def test_run_embedding_drift_score_range() -> None:
    """Embedding drift score is normalized to [0, 1]."""
    from mlops.drift_monitor import DriftMonitor

    monitor = DriftMonitor.__new__(DriftMonitor)
    ref = np.random.default_rng(0).normal(size=(50, 16))
    cur_similar = ref + 0.01
    cur_different = np.random.default_rng(1).normal(size=(50, 16))

    similar_score = monitor.run_embedding_drift(ref, cur_similar)
    different_score = monitor.run_embedding_drift(ref, cur_different)

    assert 0.0 <= similar_score <= 1.0
    assert 0.0 <= different_score <= 1.0
    assert different_score > similar_score


@pytest.mark.unit
def test_check_trajectory_drift_ks() -> None:
    """KS test detects velocity distribution shift."""
    from mlops.drift_monitor import DriftMonitor

    monitor = DriftMonitor.__new__(DriftMonitor)
    ref = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=500)
    cur = np.random.default_rng(1).normal(loc=5.0, scale=1.0, size=500)

    result = monitor.check_trajectory_drift(ref, cur)
    assert result.drift_detected
    assert result.p_value < 0.05


@pytest.mark.unit
def test_run_detection_drift_report(
    reference_parquet: Path,
    current_parquet_drifted: Path,
    tmp_path: Path,
) -> None:
    """Detection drift report flags dataset drift and writes HTML."""
    pytest.importorskip("evidently")
    from mlops.drift_monitor import DriftMonitor

    monitor = DriftMonitor(
        reference_dataset_path=reference_parquet,
        feature_columns=["confidence", "bbox_area", "class_id"],
        reports_dir=tmp_path / "reports",
    )
    current = pd.read_parquet(current_parquet_drifted)
    report = monitor.run_detection_drift_report(current)

    assert report.n_drifted_features >= 1
    assert report.share_drifted > 0.0
    assert report.report_path is not None
    assert report.report_path.exists()


@pytest.mark.unit
def test_run_detection_drift_no_drift(
    reference_parquet: Path,
    current_parquet_similar: Path,
    tmp_path: Path,
) -> None:
    """Similar current data may not trigger dataset_drift flag."""
    pytest.importorskip("evidently")
    from mlops.drift_monitor import DriftMonitor

    monitor = DriftMonitor(
        reference_dataset_path=reference_parquet,
        feature_columns=["confidence", "bbox_area", "class_id"],
        reports_dir=tmp_path / "reports",
    )
    current = pd.read_parquet(current_parquet_similar)
    report = monitor.run_detection_drift_report(current)

    assert report.share_drifted < 1.0
    assert "confidence" in report.feature_reports
