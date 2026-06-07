"""MLflow connectivity helpers with startup retry logic."""

from __future__ import annotations

import time
import urllib.error
import urllib.request

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 30
DEFAULT_DELAY_S = 2.0


def wait_for_mlflow(
    tracking_uri: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_s: float = DEFAULT_DELAY_S,
) -> bool:
    """Block until the MLflow tracking server accepts connections.

    Polls the server ``/health`` endpoint (or root) with exponential backoff
    so containerized services can wait for MLflow SQLite migrations to finish.

    Args:
        tracking_uri: MLflow tracking URI (e.g. ``http://mlflow:5000``).
        max_attempts: Maximum connection attempts before giving up.
        delay_s: Base delay in seconds between attempts.

    Returns:
        True when the server is reachable, False if all attempts failed.
    """
    base = tracking_uri.rstrip("/")
    health_urls = (f"{base}/health", base)

    for attempt in range(1, max_attempts + 1):
        for url in health_urls:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status < 500:
                        logger.info(
                            "mlflow_tracking_server_ready",
                            tracking_uri=tracking_uri,
                            attempt=attempt,
                        )
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                continue

        if attempt < max_attempts:
            sleep_s = min(delay_s * attempt, 30.0)
            logger.debug(
                "mlflow_tracking_server_unavailable",
                tracking_uri=tracking_uri,
                attempt=attempt,
                retry_in_s=sleep_s,
            )
            time.sleep(sleep_s)

    logger.warning(
        "mlflow_tracking_server_unreachable",
        tracking_uri=tracking_uri,
        max_attempts=max_attempts,
    )
    return False


def log_service_startup(
    tracking_uri: str,
    *,
    service: str,
    params: dict[str, str | int | float] | None = None,
) -> None:
    """Record a one-shot MLflow run marking service startup.

    Args:
        tracking_uri: MLflow tracking URI.
        service: Service name tag (e.g. ``ingestion`` or ``serving``).
        params: Optional parameters to log with the startup run.
    """
    if not wait_for_mlflow(tracking_uri):
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("nexus-cv")
        with mlflow.start_run(run_name=f"{service}-startup"):
            mlflow.set_tag("service", service)
            mlflow.set_tag("event", "startup")
            for key, value in (params or {}).items():
                mlflow.log_param(key, value)
        logger.info("mlflow_startup_logged", service=service, tracking_uri=tracking_uri)
    except Exception as exc:
        logger.warning("mlflow_startup_log_failed", service=service, error=str(exc))
