"""Ray actor for multi-modal sensor fusion."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import ray
import structlog

from fusion.data_types import AlignedReading, BBox3D, FusionStats, RadarTarget, Track
from fusion.kalman_tracker import MultiObjectTracker
from fusion.sensor_alignment import CameraLiDARProjector, TemporalAligner

logger = structlog.get_logger(__name__)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Compute the center of a 2D bounding box.

    Args:
        bbox: Box coordinates (x1, y1, x2, y2).

    Returns:
        Center point (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _nearest_lidar_cluster(
    track_bbox: tuple[float, float, float, float],
    aligned: AlignedReading,
    projector: CameraLiDARProjector,
) -> BBox3D | None:
    """Find the nearest LiDAR cluster to a track's 2D bbox.

    Args:
        track_bbox: Track 2D bounding box.
        aligned: Aligned multi-modal reading.
        projector: Camera-LiDAR projector for cluster matching.

    Returns:
        Nearest BBox3D cluster or None.
    """
    if not aligned.lidar.cluster_bboxes_3d:
        return None

    tcx, tcy = _bbox_center(track_bbox)
    best: BBox3D | None = None
    best_dist = float("inf")
    for bbox_3d in aligned.lidar.cluster_bboxes_3d:
        center = np.array([bbox_3d.center_xyz], dtype=np.float64)
        pixels = projector.project_lidar_to_image(center)
        px, py = pixels[0]
        dist = math.hypot(px - tcx, py - tcy)
        if dist < best_dist:
            best_dist = dist
            best = bbox_3d

    if best is not None and best_dist < 150.0:
        return best
    return None


def _nearest_radar_target(
    track_bbox: tuple[float, float, float, float],
    aligned: AlignedReading,
) -> RadarTarget | None:
    """Find the nearest radar target to a track by azimuth.

    Args:
        track_bbox: Track 2D bounding box.
        aligned: Aligned multi-modal reading.

    Returns:
        Nearest RadarTarget or None.
    """
    if not aligned.radar.targets:
        return None

    cx, _ = _bbox_center(track_bbox)
    track_azimuth = math.atan2(cx - 320.0, 554.0)
    best = None
    best_diff = float("inf")
    for target in aligned.radar.targets:
        diff = abs(target.azimuth_rad - track_azimuth)
        if diff < best_diff:
            best_diff = diff
            best = target
    if best is not None and best_diff < 0.5:
        return best
    return None


@ray.remote
class FusionActor:
    """Ray actor orchestrating multi-modal track fusion.

    Thread-safety is provided by Ray's serial actor execution model.
    """

    def __init__(
        self,
        tracker: MultiObjectTracker | None = None,
        aligner: TemporalAligner | None = None,
        projector: CameraLiDARProjector | None = None,
    ) -> None:
        """Initialize fusion components.

        Args:
            tracker: Multi-object Kalman tracker. Created if None.
            aligner: Temporal sensor aligner. Created if None.
            projector: Camera-LiDAR projector. Created if None.
        """
        self._tracker = tracker or MultiObjectTracker()
        self._aligner = aligner or TemporalAligner()
        self._projector = projector or CameraLiDARProjector()
        self._active_tracks: list[Track] = []
        logger.info("fusion_actor_initialized")

    def process_aligned_reading(self, aligned: AlignedReading) -> list[Track]:
        """Fuse an aligned multi-modal reading into enriched tracks.

        Args:
            aligned: Temporally aligned camera, LiDAR, and radar readings.

        Returns:
            Enriched track list with fused 3D and velocity data.
        """
        tracks = self._tracker.update(aligned.camera.detections)
        enriched: list[Track] = []

        for track in tracks:
            if track.state != "confirmed" or track.last_bbox_2d is None:
                enriched.append(track)
                continue

            modalities = set(track.modalities_seen)
            bbox_3d = track.last_bbox_3d
            velocity = track.velocity_2d

            cluster = _nearest_lidar_cluster(track.last_bbox_2d, aligned, self._projector)
            if cluster is not None:
                bbox_3d = cluster
                modalities.add("lidar")

            radar_target = _nearest_radar_target(track.last_bbox_2d, aligned)
            if radar_target is not None:
                velocity = (track.velocity_2d[0], radar_target.velocity_mps)
                modalities.add("radar")

            enriched_track = replace(
                track,
                last_bbox_3d=bbox_3d,
                velocity_2d=velocity,
                modalities_seen=modalities,
            )
            enriched.append(enriched_track)

        self._active_tracks = enriched
        logger.debug(
            "fusion_processed",
            num_tracks=len(enriched),
            confirmed=sum(1 for t in enriched if t.state == "confirmed"),
        )
        return enriched

    def get_active_tracks(self) -> list[Track]:
        """Return the current active fused tracks.

        Returns:
            List of active Track objects.
        """
        return list(self._active_tracks)

    def get_fusion_stats(self) -> FusionStats:
        """Compute aggregate fusion statistics.

        Returns:
            FusionStats with track counts, modality coverage, and mean age.
        """
        tracks = self._active_tracks
        state_counts: dict[str, int] = {
            "tentative": 0,
            "confirmed": 0,
            "lost": 0,
            "dead": 0,
        }
        for track in tracks:
            state_counts[track.state] = state_counts.get(track.state, 0) + 1

        total = max(len(tracks), 1)
        modality_coverage: dict[str, float] = {}
        for mod in ("camera", "lidar", "radar"):
            count = sum(1 for t in tracks if mod in t.modalities_seen)
            modality_coverage[mod] = round(100.0 * count / total, 2)

        mean_age = float(np.mean([t.age_frames for t in tracks])) if tracks else 0.0

        return FusionStats(
            track_counts_by_state=state_counts,
            modality_coverage_pct=modality_coverage,
            mean_track_age=mean_age,
        )
