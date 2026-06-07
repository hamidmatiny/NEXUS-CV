"""Shared utilities for intelligence model loading and device selection."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def select_device() -> str:
    """Select the best available compute device for inference.

    Returns:
        Device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        logger.warning("torch_unavailable", msg="PyTorch not installed; using CPU fallback")
    return "cpu"


def normalize_coords(
    cx: float,
    cy: float,
    w: float,
    h: float,
    vx: float,
    vy: float,
    img_width: float = 640.0,
    img_height: float = 480.0,
) -> tuple[float, float, float, float, float, float]:
    """Normalize bbox and velocity features to [-1, 1].

    Args:
        cx: Bounding box center x in pixels.
        cy: Bounding box center y in pixels.
        w: Bounding box width in pixels.
        h: Bounding box height in pixels.
        vx: Velocity x in pixels per frame.
        vy: Velocity y in pixels per frame.
        img_width: Reference image width.
        img_height: Reference image height.

    Returns:
        Normalized 6-tuple (cx, cy, w, h, vx, vy).
    """
    max_dim = max(img_width, img_height)
    return (
        (cx / img_width) * 2.0 - 1.0,
        (cy / img_height) * 2.0 - 1.0,
        min(w / img_width, 1.0) * 2.0 - 1.0,
        min(h / img_height, 1.0) * 2.0 - 1.0,
        max(min(vx / max_dim, 1.0), -1.0),
        max(min(vy / max_dim, 1.0), -1.0),
    )


def denormalize_coords(
    cx: float, cy: float, img_width: float = 640.0, img_height: float = 480.0
) -> tuple[float, float]:
    """Denormalize center coordinates from [-1, 1] to pixel space.

    Args:
        cx: Normalized center x.
        cy: Normalized center y.
        img_width: Reference image width.
        img_height: Reference image height.

    Returns:
        Pixel-space (cx, cy).
    """
    px = (cx + 1.0) / 2.0 * img_width
    py = (cy + 1.0) / 2.0 * img_height
    return px, py
