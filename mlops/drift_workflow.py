"""Shared drift evaluation workflow for CLI and serving pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import structlog

from mlops.drift_monitor import DriftMonitor, DriftReport

logger = structlog.get_logger(__name__)

DETECTION_FEATURE_COLUMNS = ["confidence", "bbox_area", "class_id"]


@dataclass(frozen=True, slots=True)
class DriftCheckResult:
    """Outcome of a reference-vs-current drift evaluation."""

    report: DriftReport
    exit_code: int
    current_data_path: Path | None = None

    @property
    def dataset_drift(self) -> bool:
        """Whether Evidently dataset drift was detected."""
        return self.report.dataset_drift


def build_detection_dataframe(
    confidences: list[float],
    bbox_areas: list[float],
    class_ids: list[int],
    velocities: list[float] | None = None,
) -> pd.DataFrame:
    """Build a detection feature DataFrame from accumulated observations.

    Args:
        confidences: Detection confidence scores.
        bbox_areas: Bounding box areas in pixels squared.
        class_ids: COCO class identifiers.
        velocities: Optional per-track velocity magnitudes (stored separately).

    Returns:
        DataFrame with detection feature columns.
    """
    data: dict[str, list[float] | list[int]] = {
        "confidence": confidences,
        "bbox_area": bbox_areas,
        "class_id": class_ids,
    }
    if velocities:
        data["velocity"] = velocities
    return pd.DataFrame(data)


def dump_operational_parquet(current_data: pd.DataFrame, output_dir: Path) -> Path:
    """Persist the operational inference window to a timestamped parquet file.

    Args:
        current_data: Current detection feature DataFrame.
        output_dir: Directory for operational window snapshots.

    Returns:
        Path to the written parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"current_{timestamp}.parquet"
    current_data.to_parquet(path, index=False)
    logger.info(
        "operational_window_dumped",
        path=str(path),
        rows=len(current_data),
    )
    return path


def run_drift_check(
    reference_path: Path,
    current_data: pd.DataFrame,
    reports_dir: Path,
    dataset_drift_threshold: float = 0.3,
    feature_columns: list[str] | None = None,
    current_data_path: Path | None = None,
) -> DriftCheckResult:
    """Compare operational data against reference and produce an Evidently report.

    Mirrors the logic in ``scripts/run_drift_report.py`` and returns exit code 1
    when ``dataset_drift=True``.

    Args:
        reference_path: Path to reference parquet dataset.
        current_data: Current operational feature DataFrame.
        reports_dir: Directory for HTML drift reports.
        dataset_drift_threshold: Share of drifted features constituting dataset drift.
        feature_columns: Feature columns for drift analysis.
        current_data_path: Optional pre-dumped parquet path for audit logging.

    Returns:
        DriftCheckResult with report, exit code, and data path.
    """
    columns = feature_columns or DETECTION_FEATURE_COLUMNS
    monitor = DriftMonitor(
        reference_dataset_path=reference_path,
        feature_columns=columns,
        reports_dir=reports_dir,
        dataset_drift_threshold=dataset_drift_threshold,
    )
    report = monitor.run_detection_drift_report(current_data)
    exit_code = 1 if report.dataset_drift else 0

    logger.info(
        "drift_check_complete",
        dataset_drift=report.dataset_drift,
        share_drifted=report.share_drifted,
        exit_code=exit_code,
        report_path=str(report.report_path) if report.report_path else None,
        current_data_path=str(current_data_path) if current_data_path else None,
    )
    return DriftCheckResult(
        report=report,
        exit_code=exit_code,
        current_data_path=current_data_path,
    )


def print_drift_summary(report: DriftReport) -> None:
    """Print a human-readable drift summary table to stdout.

    Args:
        report: DriftReport from Evidently analysis.
    """
    print("\n=== Drift Report Summary ===")
    print(f"{'Metric':<24} {'Value'}")
    print("-" * 40)
    print(f"{'Drifted features':<24} {report.n_drifted_features}")
    print(f"{'Share drifted':<24} {report.share_drifted:.3f}")
    print(f"{'Dataset drift':<24} {report.dataset_drift}")
    print(f"{'Report path':<24} {report.report_path}")
    print("\nPer-feature:")
    for name, details in report.feature_reports.items():
        drifted = details.get("drifted", False)
        ref_mean = details.get("reference_mean")
        cur_mean = details.get("current_mean")
        print(f"  {name}: drifted={drifted}, ref_mean={ref_mean}, cur_mean={cur_mean}")
