"""Resolve production model paths from the MLflow Model Registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

YOLO_REGISTRY_NAME = "nexus-yolo-detector"
LSTM_REGISTRY_NAME = "nexus-trajectory-lstm"


def resolve_yolo_model_path() -> str:
    """Return the production YOLO model path, falling back to settings.

    Returns:
        Local path to YOLO weights.
    """
    settings = get_settings()
    fallback = settings.YOLO_MODEL_PATH
    return _resolve_registry_path(YOLO_REGISTRY_NAME, fallback)


def resolve_lstm_model_path() -> str:
    """Return the production LSTM checkpoint path, falling back to settings.

    Returns:
        Local path to TrajectoryLSTM checkpoint.
    """
    settings = get_settings()
    fallback = settings.TRAJECTORY_LSTM_PATH
    return _resolve_registry_path(LSTM_REGISTRY_NAME, fallback)


def _resolve_registry_path(model_name: str, fallback: str) -> str:
    """Download production model from registry or return fallback path.

    Args:
        model_name: MLflow registered model name.
        fallback: Settings-based fallback path.

    Returns:
        Resolved local model path.
    """
    if not get_settings().MLFLOW_REGISTRY_ENABLED:
        return fallback

    try:
        from mlops.model_registry import ModelRegistry

        registry = ModelRegistry(tracking_uri=get_settings().MLFLOW_TRACKING_URI)
        prod = registry.get_production_model(model_name)
        if prod is None:
            logger.debug("no_production_model", model_name=model_name, fallback=fallback)
            return fallback

        cache_dir = Path(get_settings().MODEL_CACHE_DIR) / model_name / prod.version
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker = cache_dir / ".downloaded"
        if marker.exists():
            artifacts = list(cache_dir.rglob("*.pt")) + list(cache_dir.rglob("*.onnx"))
            if artifacts:
                logger.info("using_cached_model", model_name=model_name, path=str(artifacts[0]))
                return str(artifacts[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            local = registry.download_production_artifact(model_name, tmpdir)
            if local is None:
                return fallback
            src = Path(local)
            if src.is_dir():
                for artifact in src.rglob("*"):
                    if artifact.is_file() and artifact.suffix in {".pt", ".onnx", ".engine"}:
                        dest = cache_dir / artifact.name
                        dest.write_bytes(artifact.read_bytes())
                        marker.touch()
                        logger.info(
                            "production_model_resolved",
                            model_name=model_name,
                            path=str(dest),
                        )
                        return str(dest)
            elif src.is_file():
                dest = cache_dir / src.name
                dest.write_bytes(src.read_bytes())
                marker.touch()
                return str(dest)
    except Exception as exc:
        logger.warning("registry_resolve_failed", model_name=model_name, error=str(exc))

    return fallback
