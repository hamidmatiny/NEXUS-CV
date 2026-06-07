#!/usr/bin/env python3
"""Generate a drift HTML report comparing reference and current detection data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog

from mlops.drift_monitor import DriftMonitor

logger = structlog.get_logger(__name__)

FEATURE_COLUMNS = ["confidence", "bbox_area", "class_id"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument override.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Run NEXUS-CV detection drift report.")
    parser.add_argument(
        "--reference-data",
        type=Path,
        required=True,
        help="Path to reference parquet dataset",
    )
    parser.add_argument(
        "--current-data",
        type=Path,
        required=True,
        help="Path to current parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for HTML drift reports",
    )
    return parser.parse_args(argv)


def _print_summary(report: object) -> None:
    """Print drift summary table to stdout.

    Args:
        report: DriftReport instance.
    """
    from mlops.drift_monitor import DriftReport

    assert isinstance(report, DriftReport)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument override.

    Returns:
        Exit code (1 when dataset_drift=True).
    """
    args = _parse_args(argv)
    current_data = pd.read_parquet(args.current_data)

    monitor = DriftMonitor(
        reference_dataset_path=args.reference_data,
        feature_columns=FEATURE_COLUMNS,
        reports_dir=args.output_dir,
    )
    report = monitor.run_detection_drift_report(current_data)
    _print_summary(report)

    if report.dataset_drift:
        logger.warning("dataset_drift_detected", share_drifted=report.share_drifted)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
