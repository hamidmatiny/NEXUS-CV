"""Default camera-LiDAR calibration matrices for fusion projection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def default_intrinsic_matrix() -> NDArray[np.float64]:
    """Return a default 3x3 pinhole camera intrinsic matrix.

    Assumes 640x480 image, ~60° horizontal FOV, principal point at center.
    Suitable for synthetic streams; replace with calibrated values in production.

    Returns:
        3x3 intrinsic matrix K.
    """
    fx = 554.0
    fy = 554.0
    cx = 320.0
    cy = 240.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def default_extrinsic_matrix() -> NDArray[np.float64]:
    """Return a default 4x4 camera extrinsic (LiDAR → camera) transform.

    Identity rotation with LiDAR frame aligned to camera frame at origin.
    Replace with factory calibration for production deployment.

    Returns:
        4x4 homogeneous extrinsic matrix.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
