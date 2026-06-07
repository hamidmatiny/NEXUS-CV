"""Data drift monitoring using Evidently AI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray
from scipy import stats

logger = structlog.get_logger(__name__)

REPORTS_DIR = Path("reports")


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Summary of a detection data drift analysis."""

    n_drifted_features: int
    share_drifted: float
    dataset_drift: bool
    feature_reports: dict[str, dict[str, float | bool | str]]
    report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class KSDriftResult:
    """Kolmogorov-Smirnov drift test result."""

    statistic: float
    p_value: float
    drift_detected: bool


class DriftMonitor:
    """Monitors feature and embedding drift against a reference dataset."""

    def __init__(
        self,
        reference_dataset_path: Path,
        feature_columns: list[str],
        reports_dir: Path | None = None,
        dataset_drift_threshold: float = 0.3,
    ) -> None:
        """Initialize the drift monitor.

        Args:
            reference_dataset_path: Path to reference parquet dataset.
            feature_columns: Feature column names for drift analysis.
            reports_dir: Directory for HTML drift reports.
            dataset_drift_threshold: Share of drifted features for dataset_drift flag.
        """
        self._reference_path = reference_dataset_path
        self._feature_columns = feature_columns
        self._reports_dir = reports_dir or REPORTS_DIR
        self._dataset_drift_threshold = dataset_drift_threshold
        self._reference_df = pd.read_parquet(reference_dataset_path)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "drift_monitor_initialized",
            reference=str(reference_dataset_path),
            features=feature_columns,
        )

    @property
    def reference_data(self) -> pd.DataFrame:
        """Return the loaded reference dataset."""
        return self._reference_df

    def run_detection_drift_report(self, current_data: pd.DataFrame) -> DriftReport:
        """Run Evidently data drift and quality presets on detection features.

        Args:
            current_data: Current detection feature DataFrame.

        Returns:
            DriftReport summarizing drift findings.
        """
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        columns = [c for c in self._feature_columns if c in current_data.columns]
        ref_subset = self._reference_df[columns]
        cur_subset = current_data[columns]

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=ref_subset, current_data=cur_subset)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        report_path = self._reports_dir / f"drift_{timestamp}.html"
        report.save_html(str(report_path))

        drifted_features: list[str] = []
        feature_reports: dict[str, dict[str, float | bool | str]] = {}

        for col in columns:
            col_drifted = self._column_drifted(ref_subset[col], cur_subset[col])
            if col_drifted:
                drifted_features.append(col)
            feature_reports[col] = {
                "drifted": col_drifted,
                "reference_mean": float(ref_subset[col].mean()),
                "current_mean": float(cur_subset[col].mean()),
            }

        n_drifted = len(drifted_features)
        share_drifted = n_drifted / max(len(columns), 1)
        dataset_drift = share_drifted >= self._dataset_drift_threshold

        result = DriftReport(
            n_drifted_features=n_drifted,
            share_drifted=share_drifted,
            dataset_drift=dataset_drift,
            feature_reports=feature_reports,
            report_path=report_path,
        )
        logger.info(
            "detection_drift_report_complete",
            n_drifted=n_drifted,
            share_drifted=share_drifted,
            dataset_drift=dataset_drift,
            report_path=str(report_path),
        )
        return result

    def run_embedding_drift(
        self,
        reference_embeddings: NDArray[np.floating],
        current_embeddings: NDArray[np.floating],
    ) -> float:
        """Compute mean cosine-distance drift score for ViT embeddings.

        Args:
            reference_embeddings: Reference embedding matrix (N, D).
            current_embeddings: Current embedding matrix (M, D).

        Returns:
            Drift score in [0, 1] where 1 indicates maximum drift.
        """
        ref_mean = reference_embeddings.mean(axis=0)
        cur_mean = current_embeddings.mean(axis=0)

        ref_norm = np.linalg.norm(ref_mean)
        cur_norm = np.linalg.norm(cur_mean)
        if ref_norm == 0 or cur_norm == 0:
            return 0.0

        cosine_sim = float(np.dot(ref_mean, cur_mean) / (ref_norm * cur_norm))
        drift_score = float(np.clip((1.0 - cosine_sim) / 2.0, 0.0, 1.0))
        logger.info("embedding_drift_score", drift_score=drift_score, cosine_sim=cosine_sim)
        return drift_score

    def check_trajectory_drift(
        self,
        reference_velocities: NDArray[np.floating],
        current_velocities: NDArray[np.floating],
    ) -> KSDriftResult:
        """Run KS test on velocity distributions.

        Args:
            reference_velocities: Reference velocity samples.
            current_velocities: Current velocity samples.

        Returns:
            KSDriftResult with statistic, p_value, and drift flag.
        """
        statistic, p_value = stats.ks_2samp(reference_velocities, current_velocities)
        drift_detected = p_value < 0.05
        result = KSDriftResult(
            statistic=float(statistic),
            p_value=float(p_value),
            drift_detected=drift_detected,
        )
        logger.info(
            "trajectory_drift_ks_test",
            statistic=result.statistic,
            p_value=result.p_value,
            drift_detected=drift_detected,
        )
        return result

    def _column_drifted(self, reference: pd.Series, current: pd.Series) -> bool:
        """Detect per-column drift using KS test for numeric columns.

        Args:
            reference: Reference column values.
            current: Current column values.

        Returns:
            True if drift is detected.
        """
        if reference.dtype.kind not in "biufc" or current.dtype.kind not in "biufc":
            ref_mode = reference.mode().iloc[0] if len(reference) else None
            cur_mode = current.mode().iloc[0] if len(current) else None
            return ref_mode != cur_mode
        _, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
        return p_value < 0.05

    def summary_dict(self, report: DriftReport) -> dict[str, object]:
        """Convert a DriftReport to a JSON-serializable summary.

        Args:
            report: Drift report instance.

        Returns:
            Summary dictionary.
        """
        return {
            "n_drifted_features": report.n_drifted_features,
            "share_drifted": report.share_drifted,
            "dataset_drift": report.dataset_drift,
            "feature_reports": report.feature_reports,
            "report_path": str(report.report_path) if report.report_path else None,
        }

    def load_report_json(self, report_path: Path) -> dict[str, object]:
        """Load Evidently JSON export if available (helper for tests).

        Args:
            report_path: Path to report file.

        Returns:
            Parsed JSON dict or empty dict.
        """
        json_path = report_path.with_suffix(".json")
        if json_path.exists():
            return json.loads(json_path.read_text())
        return {}
