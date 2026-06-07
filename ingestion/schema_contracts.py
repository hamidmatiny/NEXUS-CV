"""Pandera schema contracts for detection validation and quarantine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera as pa
import structlog
from pandera.typing import Series

from config.settings import get_settings
from ingestion.yolo_detector import Detection

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of a detection validation run.

    Attributes:
        is_valid: True when all detections passed schema validation.
        passed: Count of detections that passed validation.
        failed: Count of detections that failed validation.
        quarantine_path: Path to quarantined parquet file, if any failures.
    """

    is_valid: bool
    passed: int
    failed: int
    quarantine_path: Path | None = None


class _DetectionRecordSchema(pa.DataFrameModel):
    """Internal Pandera model for detection row validation."""

    camera_id: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    timestamp_ns: Series[int] = pa.Field(nullable=False, gt=0)
    confidence: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0)
    bbox_x1: Series[float] = pa.Field(nullable=False, ge=0.0)
    bbox_y1: Series[float] = pa.Field(nullable=False, ge=0.0)
    bbox_x2: Series[float] = pa.Field(nullable=False, ge=0.0)
    bbox_y2: Series[float] = pa.Field(nullable=False, ge=0.0)
    class_id: Series[int] = pa.Field(nullable=False, ge=0)

    class Config:
        """Pandera model configuration."""

        coerce = True
        strict = True


detection_schema = _DetectionRecordSchema.to_schema()


def _detections_to_dataframe(
    detections: list[Detection],
    camera_id: str,
    timestamp_ns: int,
) -> pd.DataFrame:
    """Convert detections to a validation DataFrame.

    Args:
        detections: List of Detection objects.
        camera_id: Source camera identifier.
        timestamp_ns: Frame timestamp in nanoseconds.

    Returns:
        DataFrame with one row per detection.
    """
    if not detections:
        return pd.DataFrame(
            columns=[
                "camera_id",
                "timestamp_ns",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "class_id",
            ]
        )

    rows = [
        {
            "camera_id": camera_id,
            "timestamp_ns": timestamp_ns,
            "confidence": d.confidence,
            "bbox_x1": d.bbox_xyxy[0],
            "bbox_y1": d.bbox_xyxy[1],
            "bbox_x2": d.bbox_xyxy[2],
            "bbox_y2": d.bbox_xyxy[3],
            "class_id": d.class_id,
        }
        for d in detections
    ]
    return pd.DataFrame(rows)


def validate_detections(
    detections: list[Detection],
    camera_id: str,
    timestamp_ns: int | None = None,
) -> ValidationResult:
    """Validate detections against the Pandera schema contract.

    Failed rows are serialized to ``QUARANTINE_DIR/YYYYMMDD_HHMMSS_{camera_id}.parquet``.

    Args:
        detections: Detections to validate.
        camera_id: Source camera identifier.
        timestamp_ns: Frame timestamp; defaults to current time if omitted.

    Returns:
        ValidationResult summarizing pass/fail counts and quarantine path.
    """
    settings = get_settings()
    ts = timestamp_ns if timestamp_ns is not None else int(datetime.now(UTC).timestamp() * 1e9)

    if not detections:
        return ValidationResult(is_valid=True, passed=0, failed=0)

    df = _detections_to_dataframe(detections, camera_id, ts)
    passed_mask = pd.Series([True] * len(df), index=df.index)
    failed_rows: list[pd.Series] = []

    for idx, row in df.iterrows():
        row_df = pd.DataFrame([row])
        try:
            detection_schema.validate(row_df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            passed_mask.loc[idx] = False
            failed_rows.append(row)
            logger.warning(
                "detection_validation_failed",
                camera_id=camera_id,
                row_index=idx,
                errors=str(exc.failure_cases),
            )

    passed = int(passed_mask.sum())
    failed = len(df) - passed
    quarantine_path: Path | None = None

    if failed > 0:
        settings.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        quarantine_path = settings.QUARANTINE_DIR / f"{stamp}_{camera_id}.parquet"
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_parquet(quarantine_path, index=False)
        logger.info(
            "detections_quarantined",
            camera_id=camera_id,
            failed_count=failed,
            quarantine_path=str(quarantine_path),
        )

    return ValidationResult(
        is_valid=failed == 0,
        passed=passed,
        failed=failed,
        quarantine_path=quarantine_path,
    )
