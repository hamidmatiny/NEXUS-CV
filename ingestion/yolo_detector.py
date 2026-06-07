"""YOLO-based object detection with batch inference support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from config.settings import get_settings

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = structlog.get_logger(__name__)

COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result.

    Attributes:
        bbox_xyxy: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detection confidence score in [0, 1].
        class_id: Integer class identifier from the model.
        class_name: Human-readable class label.
        track_id: Optional tracking identifier when tracking is enabled.
    """

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


class YOLODetector:
    """Wraps Ultralytics YOLO for single and batch inference.

    Supports standard PyTorch weights (``.pt``) and TensorRT engines
    (``.engine``) via ``settings.YOLO_MODEL_PATH``.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Load the YOLO model.

        Args:
            model_path: Override path to weights. Defaults to
                ``settings.YOLO_MODEL_PATH``, falling back to ``yolo11n.pt``.
        """
        settings = get_settings()
        self._model_path = model_path or settings.YOLO_MODEL_PATH
        self._conf_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self._iou_threshold = settings.YOLO_IOU_THRESHOLD
        self._model: YOLO | None = None

    def _ensure_model(self) -> YOLO:
        """Lazily load the YOLO model on first use.

        Returns:
            Loaded Ultralytics YOLO instance.
        """
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for YOLO detection. Install with: pip install ultralytics"
                ) from exc

            resolved = self._resolve_model_path(self._model_path)
            logger.info("yolo_model_loading", model_path=str(resolved))
            self._model = YOLO(str(resolved))
        return self._model

    def _resolve_model_path(self, path: str) -> Path:
        """Resolve model path, falling back to yolo11n.pt when missing.

        Args:
            path: Configured model path.

        Returns:
            Path to an existing model file or default weights name.
        """
        candidate = Path(path)
        if candidate.exists():
            return candidate
        logger.warning(
            "yolo_model_fallback",
            configured=path,
            fallback="yolo11n.pt",
        )
        return Path("yolo11n.pt")

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run inference on a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of Detection objects above the confidence threshold.
        """
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        """Run batch inference on multiple frames.

        Args:
            frames: List of BGR images.

        Returns:
            Per-frame lists of Detection objects.
        """
        if not frames:
            return []

        start = time.perf_counter()
        model = self._ensure_model()
        results = model.predict(
            frames,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        batch_detections: list[list[Detection]] = []
        for result in results:
            detections: list[Detection] = []
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    track_id: int | None = None
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                    detections.append(
                        Detection(
                            bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            track_id=track_id,
                        )
                    )
            batch_detections.append(detections)

        logger.info(
            "yolo_inference_complete",
            batch_size=len(frames),
            inference_ms=round(elapsed_ms, 2),
            total_detections=sum(len(d) for d in batch_detections),
        )
        return batch_detections
