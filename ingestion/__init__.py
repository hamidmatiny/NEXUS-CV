"""Ingestion pipeline package for NEXUS-CV."""

from ingestion.frame_buffer_actor import BufferedFrame, FrameBufferActor
from ingestion.schema_contracts import ValidationResult, validate_detections
from ingestion.stream_capture import FramePacket, StreamCapture
from ingestion.yolo_detector import Detection, YOLODetector

__all__ = [
    "BufferedFrame",
    "Detection",
    "FrameBufferActor",
    "FramePacket",
    "StreamCapture",
    "ValidationResult",
    "YOLODetector",
    "validate_detections",
]
