#!/usr/bin/env python3
"""Generate a drift HTML report comparing reference and current detection data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import structlog

from mlops.drift_workflow import (
    DETECTION_FEATURE_COLUMNS,
    print_drift_summary,
    run_drift_check,
)

logger = structlog.get_logger(__name__)


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
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.3,
        help="Share of drifted features constituting dataset drift",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument override.

    Returns:
        Exit code (1 when dataset_drift=True).
    """
    args = _parse_args(argv)
    current_data = pd.read_parquet(args.current_data)

    result = run_drift_check(
        reference_path=args.reference_data,
        current_data=current_data,
        reports_dir=args.output_dir,
        dataset_drift_threshold=args.drift_threshold,
        feature_columns=DETECTION_FEATURE_COLUMNS,
        current_data_path=args.current_data,
    )
    print_drift_summary(result.report)

    if result.dataset_drift:
        logger.warning(
            "dataset_drift_detected",
            share_drifted=result.report.share_drifted,
            report_path=str(result.report.report_path),
        )
        return result.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
