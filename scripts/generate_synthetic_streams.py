#!/usr/bin/env python3
"""Generate synthetic MP4 video streams for CI and local development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 15


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list override for testing.

    Returns:
        Parsed namespace with generation parameters.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic MP4 streams with moving colored rectangles.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=4,
        help="Number of synthetic camera streams to generate (default: 4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second for output videos (default: 15)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=10,
        help="Duration of each video in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/synthetic_streams"),
        help="Directory for output MP4 files (default: ./data/synthetic_streams)",
    )
    return parser.parse_args(argv)


def _generate_frame(
    frame_idx: int,
    camera_idx: int,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a single synthetic frame with a moving rectangle.

    Args:
        frame_idx: Current frame index for animation.
        camera_idx: Camera index for color variation.
        width: Frame width in pixels.
        height: Frame height in pixels.
        rng: Random number generator.

    Returns:
        BGR image as numpy array.
    """
    frame = np.full((height, width, 3), (25, 25, 35), dtype=np.uint8)

    colors = [
        (0, 120, 255),
        (0, 200, 100),
        (200, 100, 50),
        (180, 50, 180),
        (50, 180, 220),
        (220, 180, 50),
    ]
    color = colors[camera_idx % len(colors)]

    cx = int((width // 2) + 150 * np.sin(frame_idx * 0.08 + camera_idx))
    cy = int((height // 2) + 80 * np.cos(frame_idx * 0.06 + camera_idx * 0.5))
    half = 30 + camera_idx * 5

    cv2.rectangle(
        frame,
        (cx - half, cy - half),
        (cx + half, cy + half),
        color,
        -1,
    )

    noise = rng.integers(0, 10, frame.shape, dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_streams(
    num_cameras: int,
    fps: int,
    duration_seconds: int,
    output_dir: Path,
) -> list[Path]:
    """Generate synthetic MP4 files for each camera.

    Args:
        num_cameras: Number of video files to create.
        fps: Target frames per second.
        duration_seconds: Length of each video.
        output_dir: Destination directory.

    Returns:
        List of paths to generated MP4 files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = fps * duration_seconds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    generated: list[Path] = []

    for cam_idx in range(num_cameras):
        output_path = output_dir / f"camera_{cam_idx:02d}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(fps),
            (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        )
        rng = np.random.default_rng(seed=cam_idx + 42)

        for frame_idx in range(total_frames):
            frame = _generate_frame(frame_idx, cam_idx, DEFAULT_WIDTH, DEFAULT_HEIGHT, rng)
            writer.write(frame)

        writer.release()
        generated.append(output_path)
        logger.info(
            "synthetic_stream_generated",
            camera_idx=cam_idx,
            path=str(output_path),
            frames=total_frames,
            fps=fps,
        )

    return generated


def main(argv: list[str] | None = None) -> int:
    """Entry point for the synthetic stream generator CLI.

    Args:
        argv: Optional argument list override.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = generate_streams(
        num_cameras=args.num_cameras,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        output_dir=args.output_dir,
    )
    logger.info("generation_complete", count=len(paths), output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
