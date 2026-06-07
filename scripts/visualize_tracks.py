#!/usr/bin/env python3
"""Visualize fused tracks on synthetic camera streams."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import cv2
import numpy as np
import ray
import structlog

from config.settings import get_settings
from fusion.data_types import AlignedReading, CameraDetectionReading
from fusion.fusion_actor import FusionActor
from fusion.kalman_tracker import MultiObjectTracker
from fusion.lidar_simulator import LiDARSimulator
from fusion.radar_simulator import RadarSimulator
from fusion.sensor_alignment import CameraLiDARProjector, TemporalAligner
from ingestion.stream_capture import StreamCapture
from ingestion.yolo_detector import Detection, YOLODetector

logger = structlog.get_logger(__name__)

TRACK_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument override for testing.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Visualize fused tracks on synthetic video.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./data/output/tracks_visualization.mp4"),
        help="Output annotated video path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_00",
        help="Camera identifier",
    )
    return parser.parse_args(argv)


def _color_for_track(track_id: str) -> tuple[int, int, int]:
    """Deterministic color for a track ID.

    Args:
        track_id: Track UUID string.

    Returns:
        BGR color tuple.
    """
    idx = hash(track_id) % len(TRACK_COLORS)
    return TRACK_COLORS[idx]


def _draw_tracks(
    frame: np.ndarray,
    tracks: list,
) -> np.ndarray:
    """Draw confirmed track bboxes, IDs, and velocity arrows.

    Args:
        frame: BGR image frame.
        tracks: List of Track objects.

    Returns:
        Annotated frame copy.
    """
    annotated = frame.copy()
    for track in tracks:
        if track.state != "confirmed" or track.last_bbox_2d is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in track.last_bbox_2d]
        color = _color_for_track(track.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = track.track_id[:8]
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        vx = int(track.velocity_2d[0] * 5)
        vy = int(track.velocity_2d[1] * 5)
        cv2.arrowedLine(annotated, (cx, cy), (cx + vx, cy + vy), color, 2, tipLength=0.3)
    return annotated


async def _process_frames(
    camera_id: str,
    num_frames: int,
    fusion_actor: ray.actor.ActorHandle,
    detector: YOLODetector,
    lidar_sim: LiDARSimulator,
    radar_sim: RadarSimulator,
) -> list[tuple[np.ndarray, list]]:
    """Process synthetic frames through detection and fusion.

    Args:
        camera_id: Camera identifier.
        num_frames: Frames to process.
        fusion_actor: Ray FusionActor handle.
        detector: YOLO detector.
        lidar_sim: LiDAR simulator.
        radar_sim: Radar simulator.

    Returns:
        List of (frame, tracks) tuples.
    """
    capture = StreamCapture(source_uri=f"synthetic://{camera_id}")
    results: list[tuple[np.ndarray, list]] = []
    prev_dets: list[Detection] | None = None
    count = 0

    async for packet in capture.read_frames(camera_id):
        if count >= num_frames:
            break

        detections = await asyncio.to_thread(detector.detect, packet.frame)
        ts = packet.timestamp_ns
        frame_shape = packet.frame.shape

        camera = CameraDetectionReading(
            sensor_id=camera_id,
            timestamp_ns=ts,
            detections=detections,
        )
        lidar = lidar_sim.generate(detections, frame_shape, ts)
        radar = radar_sim.generate(detections, frame_shape, ts, prev_detections=prev_dets)
        aligned = AlignedReading(
            camera=camera,
            lidar=lidar,
            radar=radar,
            alignment_gap_ms=0.0,
        )
        tracks = await asyncio.to_thread(
            ray.get,
            fusion_actor.process_aligned_reading.remote(aligned),
        )
        results.append((packet.frame, tracks))
        prev_dets = detections
        count += 1

    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point for track visualization CLI.

    Args:
        argv: Optional argument override.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args(argv)
    settings = get_settings()

    if not ray.is_initialized():
        ray.init(
            num_cpus=settings.RAY_NUM_CPUS,
            num_gpus=settings.RAY_NUM_GPUS,
            logging_level="ERROR",
        )

    fusion_actor = FusionActor.remote(
        tracker=MultiObjectTracker(min_hits=2),
        aligner=TemporalAligner(),
        projector=CameraLiDARProjector(),
    )
    detector = YOLODetector()
    lidar_sim = LiDARSimulator()
    radar_sim = RadarSimulator()

    logger.info(
        "visualize_tracks_start",
        camera_id=args.camera_id,
        num_frames=args.num_frames,
        output=str(args.output_path),
    )

    frames_and_tracks = asyncio.run(
        _process_frames(
            args.camera_id,
            args.num_frames,
            fusion_actor,
            detector,
            lidar_sim,
            radar_sim,
        )
    )

    if not frames_and_tracks:
        logger.error("no_frames_processed")
        return 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_and_tracks[0][0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output_path), fourcc, 15.0, (w, h))

    for frame, tracks in frames_and_tracks:
        annotated = _draw_tracks(frame, tracks)
        writer.write(annotated)

    writer.release()
    logger.info(
        "visualize_tracks_complete",
        output=str(args.output_path),
        frames=len(frames_and_tracks),
    )
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
