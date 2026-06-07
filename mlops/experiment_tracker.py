"""MLflow experiment tracking for NEXUS-CV model training and benchmarks."""

from __future__ import annotations

import platform
import subprocess
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


def _git_commit() -> str:
    """Return the current git commit hash or 'unknown'.

    Returns:
        Short commit hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _ray_version() -> str:
    """Return the installed Ray version or 'unknown'.

    Returns:
        Ray version string.
    """
    try:
        import ray

        return ray.__version__
    except ImportError:
        return "unknown"


class MLflowRun(AbstractContextManager["MLflowRun"]):
    """Context manager wrapping an active MLflow run."""

    def __init__(self, run: Any) -> None:
        """Initialize with an MLflow run object.

        Args:
            run: Active MLflow run from ``mlflow.start_run``.
        """
        self._run = run
        self.run_id: str = str(run.info.run_id)

    def __enter__(self) -> MLflowRun:
        """Enter the run context."""
        return self

    def __exit__(self, *args: object) -> None:
        """End the MLflow run."""
        import mlflow

        mlflow.end_run()


class NexusExperimentTracker:
    """Wraps MLflow for structured NEXUS-CV experiment logging."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "nexus-cv") -> None:
        """Initialize the experiment tracker.

        Args:
            tracking_uri: Optional MLflow tracking URI override.
            experiment_name: MLflow experiment name.
        """
        import mlflow

        from mlops.mlflow_utils import wait_for_mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            wait_for_mlflow(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name
        logger.info("experiment_tracker_initialized", experiment=experiment_name)

    def start_run(self, model_name: str, run_name: str) -> MLflowRun:
        """Start a new MLflow run as a context manager.

        Args:
            model_name: Logical model name for tagging.
            run_name: Human-readable run name.

        Returns:
            MLflowRun context manager.
        """
        import mlflow

        run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags(
            {
                "model_name": model_name,
                "git_commit": _git_commit(),
                "python_version": platform.python_version(),
                "ray_version": _ray_version(),
            }
        )
        logger.info("mlflow_run_started", run_name=run_name, model_name=model_name)
        return MLflowRun(run)

    def log_detection_run(
        self,
        model_path: Path | str,
        val_metrics: dict[str, float],
        confusion_matrix: NDArray[np.floating],
        sample_frames: list[NDArray[np.uint8]],
    ) -> None:
        """Log a YOLO detection training/evaluation run.

        Args:
            model_path: Path to the trained model weights.
            val_metrics: Validation metrics dict (mAP50, mAP50-95, etc.).
            confusion_matrix: Confusion matrix array.
            sample_frames: Sample BGR frames for artifact logging.
        """
        import mlflow

        metrics = {
            "mAP50": val_metrics.get("mAP50", 0.0),
            "mAP50-95": val_metrics.get("mAP50-95", 0.0),
            "precision": val_metrics.get("precision", 0.0),
            "recall": val_metrics.get("recall", 0.0),
            "inference_ms_p99": val_metrics.get("inference_ms_p99", 0.0),
        }
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = Path(tmpdir) / "confusion_matrix.png"
            self._save_confusion_matrix(confusion_matrix, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            model_path = Path(model_path)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")

            onnx_path = model_path.with_suffix(".onnx")
            if onnx_path.exists():
                mlflow.log_artifact(str(onnx_path), artifact_path="model")

            for idx, frame in enumerate(sample_frames[:5]):
                frame_path = Path(tmpdir) / f"sample_frame_{idx}.png"
                self._save_frame(frame, frame_path)
                mlflow.log_artifact(str(frame_path), artifact_path="samples")

        logger.info("detection_run_logged", metrics=metrics)

    def log_lstm_run(
        self,
        model_path: Path | str,
        train_loss_curve: list[float],
        val_loss_curve: list[float],
        ade_m: float,
        fde_m: float,
    ) -> None:
        """Log a TrajectoryLSTM training run.

        Args:
            model_path: Path to the trained checkpoint.
            train_loss_curve: Per-epoch training loss values.
            val_loss_curve: Per-epoch validation loss values.
            ade_m: Average displacement error in meters.
            fde_m: Final displacement error in meters.
        """
        import mlflow

        mlflow.log_metrics({"ade_m": ade_m, "fde_m": fde_m})
        mlflow.log_metric("final_train_loss", train_loss_curve[-1] if train_loss_curve else 0.0)
        mlflow.log_metric("final_val_loss", val_loss_curve[-1] if val_loss_curve else 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            loss_path = Path(tmpdir) / "loss_curves.png"
            self._save_loss_curves(train_loss_curve, val_loss_curve, loss_path)
            mlflow.log_artifact(str(loss_path), artifact_path="plots")

            path = Path(model_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="model")

        logger.info("lstm_run_logged", ade_m=ade_m, fde_m=fde_m)

    def log_system_benchmark(
        self,
        serving_percentiles: dict[str, float],
        sla_breach_rate: float,
        active_tracks_mean: float,
    ) -> None:
        """Log serving-layer benchmark metrics.

        Args:
            serving_percentiles: Latency percentiles (p50, p95, p99).
            sla_breach_rate: Fraction of requests exceeding SLA.
            active_tracks_mean: Mean active track count during benchmark.
        """
        import mlflow

        metrics = {
            "serving_p50_ms": serving_percentiles.get("p50", 0.0),
            "serving_p95_ms": serving_percentiles.get("p95", 0.0),
            "serving_p99_ms": serving_percentiles.get("p99", 0.0),
            "sla_breach_rate": sla_breach_rate,
            "active_tracks_mean": active_tracks_mean,
        }
        mlflow.log_metrics(metrics)
        logger.info("system_benchmark_logged", metrics=metrics)

    def _save_confusion_matrix(self, matrix: NDArray[np.floating], path: Path) -> None:
        """Render confusion matrix to PNG.

        Args:
            matrix: Confusion matrix array.
            path: Output PNG path.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _save_frame(self, frame: NDArray[np.uint8], path: Path) -> None:
        """Save a BGR frame as PNG.

        Args:
            frame: BGR numpy array.
            path: Output PNG path.
        """
        import cv2

        cv2.imwrite(str(path), frame)

    def _save_loss_curves(
        self,
        train_loss: list[float],
        val_loss: list[float],
        path: Path,
    ) -> None:
        """Render train/val loss curves to PNG.

        Args:
            train_loss: Training loss per epoch.
            val_loss: Validation loss per epoch.
            path: Output PNG path.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_loss, label="train")
        ax.plot(val_loss, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Training / Validation Loss")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
