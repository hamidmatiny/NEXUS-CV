"""MLflow Model Registry wrapper for NEXUS-CV models."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

DETECTION_MAP50_GATE = 0.65
LSTM_ADE_GATE_M = 1.5

STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Result of registering a model in MLflow."""

    name: str
    version: str
    run_id: str
    source: str


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """A single model registry version entry."""

    name: str
    version: str
    stage: str
    run_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Production model metadata."""

    name: str
    version: str
    stage: str
    run_id: str
    source: str


class ModelRegistry:
    """Wraps MLflow Model Registry for promotion workflows."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initialize the model registry client.

        Args:
            tracking_uri: Optional MLflow tracking URI override.
        """
        from mlflow.tracking import MlflowClient

        from mlops.mlflow_utils import wait_for_mlflow

        if tracking_uri:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            wait_for_mlflow(tracking_uri)
        self._client = MlflowClient()
        logger.info("model_registry_initialized")

    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> RegisteredModel:
        """Register a model artifact from an MLflow run.

        Args:
            run_id: Source MLflow run ID.
            model_name: Registry model name.
            artifact_path: Artifact path within the run.

        Returns:
            RegisteredModel metadata.
        """
        import mlflow

        source = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(source, model_name)
        registered = RegisteredModel(
            name=model_name,
            version=str(result.version),
            run_id=run_id,
            source=source,
        )
        logger.info(
            "model_registered",
            name=model_name,
            version=registered.version,
            run_id=run_id,
        )
        return registered

    def transition_to_staging(self, model_name: str, version: str, comment: str) -> None:
        """Transition a model version to Staging.

        Args:
            model_name: Registry model name.
            version: Model version string.
            comment: Required transition comment.
        """
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=STAGE_STAGING,
            archive_existing_versions=False,
        )
        logger.info(
            "model_transitioned_to_staging",
            model_name=model_name,
            version=version,
            comment=comment,
        )

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        validation_metrics: dict[str, float],
        force: bool = False,
    ) -> None:
        """Promote a model to Production after gate checks.

        Args:
            model_name: Registry model name.
            version: Model version string.
            validation_metrics: Metrics used for promotion gates.
            force: Skip gate checks when True.

        Raises:
            ValueError: When promotion gates are not met.
        """
        if not force:
            self._check_promotion_gates(model_name, validation_metrics)

        current_prod = self.get_production_model(model_name)
        if current_prod is not None:
            self._client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage=STAGE_ARCHIVED,
                archive_existing_versions=False,
            )
            logger.info(
                "previous_production_archived",
                model_name=model_name,
                version=current_prod.version,
            )

        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=STAGE_PRODUCTION,
            archive_existing_versions=True,
        )
        logger.info(
            "model_promoted_to_production",
            model_name=model_name,
            version=version,
            metrics=validation_metrics,
        )

    def get_production_model(self, model_name: str) -> ModelInfo | None:
        """Return the current Production model for a registered name.

        Args:
            model_name: Registry model name.

        Returns:
            ModelInfo if a Production version exists, else None.
        """
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
        except Exception:
            return None

        for mv in versions:
            if mv.current_stage == STAGE_PRODUCTION:
                return ModelInfo(
                    name=model_name,
                    version=str(mv.version),
                    stage=mv.current_stage,
                    run_id=mv.run_id,
                    source=mv.source,
                )
        return None

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions for a registered model.

        Args:
            model_name: Registry model name.

        Returns:
            List of ModelVersion entries.
        """
        versions = self._client.search_model_versions(f"name='{model_name}'")
        return [
            ModelVersion(
                name=model_name,
                version=str(mv.version),
                stage=mv.current_stage,
                run_id=mv.run_id,
                status=mv.status,
            )
            for mv in versions
        ]

    def download_production_artifact(self, model_name: str, dst_path: str) -> str | None:
        """Download the Production model artifact to a local path.

        Args:
            model_name: Registry model name.
            dst_path: Local destination directory.

        Returns:
            Local path to downloaded artifact, or None if no production model.
        """
        prod = self.get_production_model(model_name)
        if prod is None:
            return None

        local_path = self._client.download_artifacts(prod.run_id, "model", dst_path=dst_path)
        logger.info(
            "production_model_downloaded",
            model_name=model_name,
            version=prod.version,
            local_path=local_path,
        )
        return local_path

    def _check_promotion_gates(self, model_name: str, metrics: dict[str, float]) -> None:
        """Validate promotion gate metrics.

        Args:
            model_name: Registry model name.
            metrics: Validation metrics dict.

        Raises:
            ValueError: When gates fail.
        """
        name_lower = model_name.lower()
        if "lstm" in name_lower or "trajectory" in name_lower:
            ade = metrics.get("ade_m", float("inf"))
            if ade >= LSTM_ADE_GATE_M:
                raise ValueError(f"LSTM ADE gate failed: {ade:.3f}m >= {LSTM_ADE_GATE_M}m")
            return

        map50 = metrics.get("mAP50", 0.0)
        if map50 <= DETECTION_MAP50_GATE:
            raise ValueError(f"Detection mAP50 gate failed: {map50:.3f} <= {DETECTION_MAP50_GATE}")

    def evaluate_promotion(
        self,
        model_name: str,
        version: str,
        validation_metrics: dict[str, float],
        force: bool = False,
    ) -> tuple[bool, str]:
        """Evaluate whether a model can be promoted without applying changes.

        Args:
            model_name: Registry model name.
            version: Target version.
            validation_metrics: Metrics for gate checks.
            force: Skip gate checks.

        Returns:
            Tuple of (can_promote, reason message).
        """
        if force:
            return True, f"Force promotion of {model_name} v{version} (gates skipped)"
        try:
            self._check_promotion_gates(model_name, validation_metrics)
            return True, f"All promotion gates passed for {model_name} v{version}"
        except ValueError as exc:
            return False, str(exc)
